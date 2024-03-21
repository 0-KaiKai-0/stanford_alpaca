# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import Trainer


class PipeTrainer(Trainer):
    def __init__(self, alpha=0.5, output_rationale=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale


    def compute_loss(self, model, inputs, return_outputs=False):
        rationale_outputs = model(**inputs['rationale'])
        
        with torch.no_grad():
            rationale_pos = inputs['rationale']['labels'].eq(-100).sum(-1) - inputs['rationale']['input_ids'].eq(32000).sum(-1)
            rationales = rationale_outputs.logits.argmax(-1)
            for idx in range(rationales.shape[0]):
                rationale = rationales[idx][rationale_pos[idx] - 1:]
                if rationale.shape[0] > 96:
                    rationale = rationale[:96]
                inputs['output']['input_ids'][idx][1:rationale.shape[0] + 1] = rationale

        output_outputs = model(**inputs['output'])
        loss = self.alpha * rationale_outputs.loss + (1. - self.alpha) * output_outputs.loss

        return (loss, {'rationale': rationale_outputs, 'output': output_outputs}) if return_outputs else loss


class RatrionaleGuidedTrainer(Trainer):
    def __init__(self, alpha=0, output_rationale=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale


    def compute_loss(self, model, inputs, return_outputs=False):
        # with torch.no_grad():
        rationale_outputs = model(**inputs['rationale'])
        output_outputs = model(**inputs['output'])
        output_loss = rationale_outputs.loss
        
        # import pdb
        # pdb.set_trace()
        rationale_end = inputs['rationale']['labels'].shape[1] - inputs['rationale']['input_ids'].eq(32000).sum(-1)
        output_pos = inputs['output']['labels'].eq(-100).sum(-1) - inputs['output']['input_ids'].eq(32000).sum(-1)
        labels_len = inputs['output']['labels'].shape[1] - inputs['output']['labels'].eq(-100).sum(-1)
        
        rationale_probs_list = []
        output_probs_list = []
        for idx in range(inputs['output']['input_ids'].shape[0]):
            rationale_prob = nn.functional.softmax(rationale_outputs.logits[idx][rationale_end[idx]-labels_len[idx]:rationale_end[idx]], dim=-1)
            output_prob = nn.functional.softmax(output_outputs.logits[idx][output_pos[idx]:output_pos[idx]+labels_len[idx]], dim=-1)
            prob_len = min(rationale_prob.shape[0], output_prob.shape[0])
            rationale_probs_list.append(rationale_prob[:prob_len])
            output_probs_list.append(output_prob[:prob_len])
        
        rationale_probs = torch.cat(rationale_probs_list, dim=0)
        output_probs = torch.cat(output_probs_list, dim=0)
   
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        log_output_probs = torch.log(output_probs)
        log_rationale_probs = torch.log(rationale_probs)
        M = ((output_probs + rationale_probs) / 2)
        log_M = torch.log(M)
        # prob_loss = loss_fct(log_output_probs, rationale_probs)
        # Rdetach_logR_logM:
        # prob_loss = (loss_fct(log_rationale_probs, M) + loss_fct(log_M, output_probs)) / 2
        # arg2detach_logR_logM
        prob_loss = (loss_fct(log_rationale_probs, M.detach()) + loss_fct(log_M, output_probs.detach())) / 2
        # prob_loss = loss_fct(log_rationale_probs.detach(), output_probs)
        
        loss = output_loss + self.alpha * prob_loss
        # import pdb
        # pdb.set_trace()

        return (loss, {'rationale': rationale_outputs, 'output': output_outputs}) if return_outputs else loss
