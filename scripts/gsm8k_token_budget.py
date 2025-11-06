# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import random

import datasets


INSTRUCTION_FOLLOWING = "Think step by step and output the final answer using \\boxed{}."
TOKEN_BUDGET_STR = 'You have a token budget of around {budget} tokens. You must finish your thinking process within +50 or -50 tokens of the thinking budget.'


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def make_base_map_fn(split):
    def process_fn(example, idx):
        question_raw = example.pop("question")
        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)
        return {
            "question_raw": question_raw,
            "answer_raw": answer_raw,
            "solution": solution,
            "split": split,
            "index": idx,
        }
    return process_fn


def expand_with_budgets(dataset, num_copies, budget_values, data_source):
    """Create multiple copies of each example with different token budgets."""
    expanded = []
    
    for example in dataset:
        budgets = random.sample(budget_values, num_copies)
        
        for copy_idx, budget in enumerate(budgets):
            # Handle control condition (no budget constraint)
            if budget == 'control':
                question = example["question_raw"] + "\n\n" + INSTRUCTION_FOLLOWING
            else:
                question = (
                    example["question_raw"] + "\n\n" + 
                    TOKEN_BUDGET_STR.format(budget=budget) + " " + 
                    INSTRUCTION_FOLLOWING
                )
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": example["solution"]},
                "extra_info": {
                    "split": example["split"],
                    "index": example["index"],
                    "answer": example["answer_raw"],
                    "question": example["question_raw"],
                    "budget": budget,
                    "copy_idx": copy_idx,
                },
            }
            expanded.append(data)
    
    return datasets.Dataset.from_list(expanded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--num_budget_copies", type=int, default=1, help="Number of copies of each problem with different budgets."
    )
    parser.add_argument(
        "--budget_values", nargs='+', default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        help="Possible token budget values to sample from. Can include 'control' for no budget constraint."
    )

    args = parser.parse_args()
    
    num_copies = args.num_budget_copies
    # Convert budget values to int except for 'control'
    budget_values = []
    for val in args.budget_values:
        if val == 'control':
            budget_values.append('control')
        else:
            budget_values.append(int(val))
    
    if num_copies > len(budget_values):
        raise ValueError(
            f"num_budget_copies ({num_copies}) cannot exceed the number of "
            f"available budget values ({len(budget_values)})"
        )
    
    local_dataset_path = args.local_dataset_path

    data_source = "openai/gsm8k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Extract base data first
    train_dataset = train_dataset.map(function=make_base_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_base_map_fn("test"), with_indices=True)
    
    # Expand with different budgets
    train_dataset = expand_with_budgets(train_dataset, num_copies, budget_values, data_source)
    test_dataset = expand_with_budgets(test_dataset, num_copies, budget_values, data_source)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
