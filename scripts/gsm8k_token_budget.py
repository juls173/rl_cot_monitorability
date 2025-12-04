"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import random
from typing import Optional, Union

import datasets


INSTRUCTION_FOLLOWING = "Think step by step and output the final answer using \\boxed{}."
INSTRUCTION_FOLLOWING_FORMAT_ONLY = "Think step by step and output ONLY the final answer using \\boxed{}. Do not provide any other explanation."
TOKEN_BUDGET_STR = 'You have a token budget of around {budget} tokens. You must finish your thinking process as close as possible to the thinking budget.'
TOKEN_BUDGET_WINDOW_STR = 'You have a token budget of around {budget} tokens. You must finish your thinking process within +/- {window} tokens of the budget.'


def random_multiple_of_10(lo: int, hi: int) -> int:
    """Sample a random multiple of 10 within [lo, hi]."""
    lo_rounded = ((lo + 9) // 10) * 10  # Round up to nearest 10
    hi_rounded = (hi // 10) * 10  # Round down to nearest 10
    if lo_rounded > hi_rounded:
        # Range too small, just return midpoint rounded to 10
        return round((lo + hi) / 2 / 10) * 10
    return random.randint(lo_rounded // 10, hi_rounded // 10) * 10


def get_curriculum_budget(
    position: float,
    budget_values: list,
    budget_range: bool,
    curriculum_warmup: Optional[float],
    curriculum_cooldown: Optional[float],
) -> Union[int, str]:
    """
    Get budget for a given position in curriculum mode.
    
    Args:
        position: Position in dataset (0.0 to 1.0)
        budget_values: Budget values (2 for range, 4 for range+curriculum, or discrete list)
        budget_range: If True, budget_values defines a range
        curriculum_warmup: Fraction of dataset using initial budget
        curriculum_cooldown: Fraction of dataset using final budget
    
    Returns:
        Budget value (int) or 'control'
    """
    warmup = curriculum_warmup if curriculum_warmup is not None else 0.0
    cooldown = curriculum_cooldown if curriculum_cooldown is not None else 0.0
    
    if budget_range:
        # budget_values = [init_lo, init_hi, final_lo, final_hi]
        init_lo, init_hi, final_lo, final_hi = budget_values
        
        if position < warmup:
            return random_multiple_of_10(init_lo, init_hi)
        elif position >= 1.0 - cooldown:
            return random_multiple_of_10(final_lo, final_hi)
        else:
            # Linear interpolation
            middle_length = 1.0 - warmup - cooldown
            progress = (position - warmup) / middle_length if middle_length > 0 else 0.5
            lo = round(init_lo + progress * (final_lo - init_lo))
            hi = round(init_hi + progress * (final_hi - init_hi))
            return random_multiple_of_10(min(lo, hi), max(lo, hi))
    else:
        # Discrete values - sort descending for curriculum
        numeric_values = sorted([v for v in budget_values if v != 'control'], reverse=True)
        n_values = len(numeric_values)
        
        if n_values == 0:
            return 'control'
        
        if warmup == 0.0 and cooldown == 0.0:
            # Default: distribute all values evenly
            idx = int(position * n_values)
            idx = min(idx, n_values - 1)
            return numeric_values[idx]
        
        if position < warmup:
            return numeric_values[0]  # Highest
        elif position >= 1.0 - cooldown:
            return numeric_values[-1]  # Lowest
        else:
            # Middle section uses remaining values (excluding first and last)
            if n_values <= 2:
                mid = warmup + (1.0 - warmup - cooldown) / 2
                return numeric_values[0] if position < mid else numeric_values[-1]
            
            middle_values = numeric_values[1:-1]
            middle_length = 1.0 - warmup - cooldown
            progress = (position - warmup) / middle_length if middle_length > 0 else 0.5
            idx = int(progress * len(middle_values))
            idx = min(idx, len(middle_values) - 1)
            return middle_values[idx]


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


def expand_with_budgets(
    dataset,
    num_copies: int,
    budget_values: list,
    data_source: str,
    format_only_answer: bool = False,
    budget_window: int = 0,
    decreasing_budgets: bool = False,
    budget_range: bool = False,
    curriculum_warmup: Optional[float] = None,
    curriculum_cooldown: Optional[float] = None,
    is_validation: bool = False,
):
    """Create multiple copies of each example with different token budgets.
    
    Args:
        dataset: Input dataset
        num_copies: Number of copies per example
        budget_values: Budget values (discrete list, or range endpoints)
        data_source: Data source identifier
        format_only_answer: Use format-only instruction
        budget_window: Window around budget for penalty-free zone
        decreasing_budgets: Use curriculum (budgets decrease over training)
        budget_range: Interpret budget_values as range (2 values) or interpolated range (4 values with curriculum)
        curriculum_warmup: Fraction of dataset for initial budget (curriculum mode only)
        curriculum_cooldown: Fraction of dataset for final budget (curriculum mode only)
        is_validation: If True and curriculum is used, use only final budget values
    """
    expanded = []
    instruction = INSTRUCTION_FOLLOWING_FORMAT_ONLY if format_only_answer else INSTRUCTION_FOLLOWING
    
    def make_example(example, budget, copy_idx):
        """Helper to create a single expanded example."""
        if budget == 'control':
            question = example["question_raw"] + "\n\n" + instruction
        else:
            if budget_window > 0:
                budget_str = TOKEN_BUDGET_WINDOW_STR.format(budget=budget, window=budget_window)
            else:
                budget_str = TOKEN_BUDGET_STR.format(budget=budget)
            question = example["question_raw"] + "\n\n" + budget_str + " " + instruction
        
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
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
    
    if decreasing_budgets:
        # Curriculum mode: assign budgets based on position
        # For validation, use only final budget values
        if is_validation:
            for example in dataset:
                for copy_idx in range(num_copies):
                    if budget_range:
                        # Use final range [final_lo, final_hi]
                        final_lo, final_hi = budget_values[2], budget_values[3]
                        budget = random_multiple_of_10(final_lo, final_hi)
                    else:
                        # Use lowest (final) budget value
                        numeric_values = sorted([v for v in budget_values if v != 'control'])
                        budget = numeric_values[0] if numeric_values else 'control'
                    expanded.append(make_example(example, budget, copy_idx))
        else:
            total_examples = len(dataset) * num_copies
            global_idx = 0
            
            for example in dataset:
                for copy_idx in range(num_copies):
                    position = global_idx / total_examples if total_examples > 0 else 0
                    budget = get_curriculum_budget(
                        position, budget_values, budget_range,
                        curriculum_warmup, curriculum_cooldown
                    )
                    expanded.append(make_example(example, budget, copy_idx))
                    global_idx += 1
    else:
        # Random sampling mode
        for example in dataset:
            if budget_range:
                # Sample from continuous range [min, max] (multiples of 10)
                budgets = [random_multiple_of_10(budget_values[0], budget_values[1]) for _ in range(num_copies)]
            else:
                # Sample from discrete values (without replacement)
                budgets = random.sample(budget_values, num_copies)
            
            for copy_idx, budget in enumerate(budgets):
                expanded.append(make_example(example, budget, copy_idx))
    
    return datasets.Dataset.from_list(expanded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default=None, help="The save directory for the preprocessed dataset.")
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
    parser.add_argument(
        "--budget-range",
        action="store_true",
        help="Interpret budget values as a range (2 values) or interpolated range (4 values with curriculum)"
    )
    parser.add_argument(
        "--curriculum-warmup",
        type=float,
        default=None,
        help="Fraction of dataset using initial budget (curriculum mode only)"
    )
    parser.add_argument(
        "--curriculum-cooldown",
        type=float,
        default=None,
        help="Fraction of dataset using final budget (curriculum mode only)"
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
    
    # Validation
    if args.budget_range:
        if args.decreasing_budgets:
            if len(budget_values) != 4:
                raise ValueError(
                    "--budget-range with --decreasing-budgets requires exactly 4 values: "
                    "init_lo, init_hi, final_lo, final_hi"
                )
        else:
            if len(budget_values) != 2:
                raise ValueError("--budget-range requires exactly 2 values: min, max")
    else:
        if num_copies > len(budget_values):
            raise ValueError(
                f"num_budget_copies ({num_copies}) cannot exceed the number of "
                f"available budget values ({len(budget_values)})"
            )
        if args.decreasing_budgets and len(budget_values) < 2:
            raise ValueError("--decreasing-budgets requires at least 2 budget values")
    
    if (args.curriculum_warmup is not None or args.curriculum_cooldown is not None) and not args.decreasing_budgets:
        raise ValueError("--curriculum-warmup and --curriculum-cooldown require --decreasing-budgets")
    
    if args.curriculum_warmup is not None and args.curriculum_cooldown is not None:
        if args.curriculum_warmup + args.curriculum_cooldown > 1.0:
            raise ValueError("curriculum_warmup + curriculum_cooldown cannot exceed 1.0")
    
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
    train_dataset = expand_with_budgets(
        train_dataset, num_copies, budget_values, data_source,
        args.format_only_answer, args.budget_window, args.decreasing_budgets,
        args.budget_range, args.curriculum_warmup, args.curriculum_cooldown,
        is_validation=False
    )
    test_dataset = expand_with_budgets(
        test_dataset, num_copies, budget_values, data_source,
        args.format_only_answer, args.budget_window, args.decreasing_budgets,
        args.budget_range, args.curriculum_warmup, args.curriculum_cooldown,
        is_validation=True
    )

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local-dir' is deprecated. Please use 'local-save-dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
