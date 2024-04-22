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
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import Trainer


def WeightedMSE(student_logits, teacher_logits, labels):
    k = 1
    p = F.log_softmax(teacher_logits, 1)
    ohe_labels = F.one_hot(labels, num_classes=teacher_logits.shape[1])
    nn_l = torch.sum(ohe_labels.float() * p, dim=1)
    weight = torch.exp(k * nn_l)
    return torch.mean(weight * torch.sum((student_logits - teacher_logits) ** 2, dim=1))
        
def KLD(student_logits, teacher_logits, labels):
    # import pdb
    # pdb.set_trace()
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    inf_mask = torch.isinf(student_logits)
    logprobs = F.log_softmax(student_logits, dim=-1)
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    loss_mask = (labels != -100).int()
    distil_loss = -torch.sum(x * loss_mask.view(-1), dim=0) / torch.sum(loss_mask.view(-1), dim=0)
    return distil_loss

def ReverseKLD(student_logits, teacher_logits, labels):
    student_probs = F.softmax(student_logits, dim=-1)
    student_logprobs = F.log_softmax(student_logits, dim=-1)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(student_logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (labels != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def SymmetricKLD(student_logits, teacher_logits, labels, lam=0.3):
    for_kl = KLD(student_logits, teacher_logits, labels)
    rev_kl = ReverseKLD(student_logits, teacher_logits, labels)
    distil_loss = (1-lam) * for_kl + lam * rev_kl
    return distil_loss

def JS_KLD(student_logits, teacher_logits, labels, lam=0.7):
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
    student_logprobs = F.log_softmax(student_logits, dim=-1)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (labels != -100).int()
    inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1-lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def tv_distance(student_logits, teacher_logits, labels):
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)
    
    mask = (labels != -100).int()
    inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_forward_kl(student_logits, teacher_logits, labels, lam=0.7):
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)
    
    mask = (labels != -100).int()
    inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_reverse_kl(student_logits, teacher_logits, labels, lam=0.7):
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    student_probs = F.softmax(student_logits, dim=-1)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    
    student_logprobs = F.log_softmax(student_logits, dim=-1)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (labels != -100).int()
    inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
    

class PipeTrainer(Trainer):
    def __init__(self, alpha=1, output_rationale=False, **kwargs):
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
    def __init__(self, alpha=1, output_rationale=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale


    def compute_loss(self, model, inputs, return_outputs=False):
        # with torch.no_grad():
        rationale_outputs = model(**inputs['rationale'])
        rationale_loss = rationale_outputs.loss
        output_outputs = model(**inputs['output'])
        # output_loss = output_outputs.loss
        # self.tokenizer.decode(inputs['rationale']['labels'][0].tolist())
        rationale_end = inputs['rationale']['labels'].shape[1] - inputs['rationale']['input_ids'].eq(32000).sum(-1)
        output_pos = inputs['output']['labels'].eq(-100).sum(-1) - inputs['output']['input_ids'].eq(32000).sum(-1)
        labels_len = inputs['output']['labels'].shape[1] - inputs['output']['labels'].eq(-100).sum(-1)
        
        rationale_logits_list = []
        output_logits_list = []
        label_list = []
        for idx in range(inputs['output']['input_ids'].shape[0]):
            # import pdb
            # pdb.set_trace()
            rationale_logit = rationale_outputs.logits[idx][rationale_end[idx]-labels_len[idx]:rationale_end[idx]]
            output_logit = output_outputs.logits[idx][output_pos[idx]:output_pos[idx]+labels_len[idx]]
            prob_len = min(rationale_logit.shape[0], output_logit.shape[0])
            rationale_logits_list.append(rationale_logit[:prob_len])
            output_logits_list.append(output_logit[:prob_len])
            
            label = inputs['output']['labels'][idx][output_pos[idx]:output_pos[idx]+labels_len[idx]]
            label_list.append(label[:prob_len])
        
        rationale_logits = torch.cat(rationale_logits_list, dim=0)
        output_logits = torch.cat(output_logits_list, dim=0)
        labels = torch.cat(label_list, dim=0)
        
        kd_loss = skewed_reverse_kl(output_logits, rationale_logits.detach(), labels)
        loss = rationale_loss + self.alpha * kd_loss
        # import pdb
        # pdb.set_trace()
        # output_outputs = None
        # loss = rationale_loss

        return (loss, {'rationale': rationale_outputs, 'output': output_outputs}) if return_outputs else loss
