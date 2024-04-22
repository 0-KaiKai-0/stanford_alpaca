#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
# from transformers import Trainer
from model_utils import RatrionaleGuidedTrainer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
RATIONALE_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that concisely describes the instruction and the input, and appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that concisely describes the instruction, and appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
OUTPUT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
TARGET = ("{rationale}\n\n###So the answer is:\n{output}")



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)
        # list_data_dict = list_data_dict[:int(len(list_data_dict) * 0.01)]

        logging.warning("Formatting inputs...")
        rationale_input, rationale_no_input = RATIONALE_DICT["prompt_input"], RATIONALE_DICT["prompt_no_input"]
        rationale_sources = [
            rationale_input.format_map(example) if example.get("input", "") != "" else rationale_no_input.format_map(example)
            for example in list_data_dict
        ]
        rationale_targets = [
            TARGET.format_map(example) + tokenizer.eos_token if example.get("rationale", "") != "" else f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]
        
        output_input, output_no_input = OUTPUT_DICT["prompt_input"], OUTPUT_DICT["prompt_no_input"]
        output_sources = [
            output_input.format_map(example) if example.get("input", "") != "" else output_no_input.format_map(example)
            for example in list_data_dict
        ]
        output_targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        # import pdb
        # pdb.set_trace()

        logging.warning("Tokenizing inputs... This may take some time...")
        rationale_data_dict = preprocess(rationale_sources, rationale_targets, tokenizer)
        output_data_dict = preprocess(output_sources, output_targets, tokenizer)

        self.rationale_input_ids = rationale_data_dict["input_ids"]
        self.rationale_labels = rationale_data_dict["labels"]
        self.output_input_ids = output_data_dict["input_ids"]
        self.output_labels = output_data_dict["labels"]

    def __len__(self):
        return len(self.rationale_input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            rationale_input_ids=self.rationale_input_ids[i],
            rationale_labels=self.rationale_labels[i],
            output_input_ids=self.output_input_ids[i],
            output_labels=self.output_labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # import pdb
        # pdb.set_trace()
        rationale_input_ids, rationale_labels = tuple([instance[key] for instance in instances] for key in ("rationale_input_ids", "rationale_labels"))
        rationale_input_ids = torch.nn.utils.rnn.pad_sequence(
            rationale_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        rationale_labels = torch.nn.utils.rnn.pad_sequence(rationale_labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        output_input_ids, output_labels = tuple([instance[key] for instance in instances] for key in ("output_input_ids", "output_labels"))
        output_input_ids = torch.nn.utils.rnn.pad_sequence(
            output_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        output_labels = torch.nn.utils.rnn.pad_sequence(output_labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        rationale_features = dict(
            input_ids=rationale_input_ids,
            labels=rationale_labels,
            attention_mask=rationale_input_ids.ne(self.tokenizer.pad_token_id),
        )
        output_features = dict(
            input_ids=output_input_ids,
            labels=output_labels,
            attention_mask=output_input_ids.ne(self.tokenizer.pad_token_id),
        )
        return {
            'rationale': rationale_features,
            'output': output_features
        }


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = RatrionaleGuidedTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
