"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import random

import datasets


INSTRUCTION_FOLLOWING = "Think step by step and output the final answer using \\boxed{}."
INSTRUCTION_FOLLOWING_FORMAT_ONLY = "Think step by step and output ONLY the final answer using \\boxed{}. Do not provide any other explanation."
TOKEN_BUDGET_STR = 'You have a token budget of around {budget} tokens. You must finish your thinking process as close as possible to the thinking budget.'
TOKEN_BUDGET_WINDOW_STR = 'You have a token budget of around {budget} tokens. You must finish your thinking process within +/- {window} tokens of the budget.'


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


def expand_with_budgets(dataset, num_copies, budget_values, data_source, format_only_answer=False, budget_window=0, decreasing_budgets=False):
    """Create multiple copies of each example with different token budgets.
    
    If decreasing_budgets is True, examples are sorted by budget (descending) after creation.
    """
    expanded = []
    
    instruction = INSTRUCTION_FOLLOWING_FORMAT_ONLY if format_only_answer else INSTRUCTION_FOLLOWING
    
    for example in dataset:
        budgets = random.sample(budget_values, num_copies)
        
        for copy_idx, budget in enumerate(budgets):
            # Handle control condition (no budget constraint)
            if budget == 'control':
                question = example["question_raw"] + "\n\n" + instruction
            else:
                # Use window-aware prompt if budget_window > 0
                if budget_window > 0:
                    budget_str = TOKEN_BUDGET_WINDOW_STR.format(budget=budget, window=budget_window)
                else:
                    budget_str = TOKEN_BUDGET_STR.format(budget=budget)
                question = (
                    example["question_raw"] + "\n\n" + 
                    budget_str + " " + 
                    instruction
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
                    "budget_window": budget_window,
                    "copy_idx": copy_idx,
                    "format_only_answer": format_only_answer,
                },
            }
            expanded.append(data)
    
    # Sort by budget descending if requested (treat 'control' as infinity so it comes first)
    if decreasing_budgets:
        expanded.sort(key=lambda x: float('inf') if x["extra_info"]["budget"] == 'control' else x["extra_info"]["budget"], reverse=True)
    
    return datasets.Dataset.from_list(expanded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs-dir", default=None)
    parser.add_argument("--local-dataset-path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local-save-dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--num-budget-copies", type=int, default=1, help="Number of copies of each problem with different budgets."
    )
    parser.add_argument(
        "--budget-values", nargs='+', default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        help="Possible token budget values to sample from. Can include 'control' for no budget constraint."
    )
    parser.add_argument(
        "--format-only-answer",
        action="store_true",
        help="Instruct model to output only boxed answer with no explanation"
    )
    parser.add_argument(
        "--budget-window",
        type=int,
        default=0,
        help="Window around budget where no penalty is applied (default: 0)"
    )
    parser.add_argument(
        "--decreasing-budgets",
        action="store_true",
        help="Sort examples by budget (descending) for curriculum-style training"
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
    
    if args.decreasing_budgets and len(budget_values) < 2:
        raise ValueError("--decreasing_budgets requires at least 2 budget values")
    
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
    train_dataset = expand_with_budgets(train_dataset, num_copies, budget_values, data_source, args.format_only_answer, args.budget_window, args.decreasing_budgets)
    test_dataset = expand_with_budgets(test_dataset, num_copies, budget_values, data_source, args.format_only_answer, args.budget_window, args.decreasing_budgets)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local-dir' is deprecated. Please use 'local-save-dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
