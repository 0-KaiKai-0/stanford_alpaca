import openai
import os
import json
import argparse
import logging
import time
import utils
from tenacity import (
    retry,
    wait_random_exponential
)
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--key', type=str)
args = parser.parse_args()

openai.api_key = args.key

stop_after_attempt = 8
attempts = 0

@retry(wait=wait_random_exponential(min=1, max=20))
def get_rationales(system_intel, prompt, n=1):
    global attempts
    attempts += 1
    print(attempts)
    if attempts > stop_after_attempt:
        print("failed to generate rationale")
        attempts = 0
        return ""
    result = openai.ChatCompletion.create(model=args.model,
                                        messages=[{"role": "system", "content": system_intel},
                                            {"role": "user", "content": prompt}],
                                        temperature=0,
                                        max_tokens=100,
                                        n=n)
    rationale = result['choices'][0]['message']['content']
    attempts = 0
    return rationale


logging.warning("Loading data...")
list_data_dict = utils.jload(args.data_path)

logging.warning("Formatting inputs...")
# import pdb
# pdb.set_trace()

system_intels = ["You are an encyclopaedia. You are provided with an instruction that describes a task. Do not answer it directly. Please depict this instruction and provide relevant factual information in real world as much as possible. Your description should begin with 'This instruction is asking for' and can not be more than 80 words.",
                 "You are an encyclopaedia. You are provided with an instruction that describes a task, paired with an input that provides further context. Do not answer it directly. Please depict this instruction and the following input, and provide relevant factual information in real world as much as possible. Your description should begin with 'This instruction is asking for' and can not be more than 90 words."]

# system_intels = ["You are an encyclopaedia. You are provided with an instruction that describes a task. Please depict key words in this instruction and provide relevant factual information in real world as much as possible. Your description can not be more than 100 words.",
#                  "You are an encyclopaedia. You are provided with an instruction that describes a task, paired with an input that provides further context. Please depict key words in this instruction and the following input, and provide relevant factual information in real world as much as possible. Your description can not be more than 100 words."]
for idx in tqdm(range(len(list_data_dict))):    
    sample = list_data_dict[idx]
    if sample.get("input", "") == "":
        system_intel = system_intels[0]
        prompt = f"Instruction:\n{sample['instruction']}"
    else:
        system_intel = system_intels[1]
        prompt = f"Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}"
    rationale = get_rationales(system_intel, prompt)
    print("-"*20 + str(idx) + "-"*20)
    print(system_intel)
    print(prompt)
    print(rationale)
    list_data_dict[idx]["rationale"] = rationale
utils.jdump(list_data_dict, args.output_path)

