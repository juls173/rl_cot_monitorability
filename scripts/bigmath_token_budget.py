"""
Preprocess the Big-Math-RL-Verified dataset to parquet format with token budgets
"""

import argparse
import os
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


def is_numerical(answer: str) -> bool:
    """Check if an answer is numerical."""
    # Remove commas and whitespace
    cleaned = answer.replace(",", "").strip()
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def make_base_map_fn(split):
    def process_fn(example, idx):
        problem = example.pop("problem")
        answer = example.pop("answer")
        source = example.pop("source")
        domain = example.pop("domain")
        solve_rate = example.pop("llama8b_solve_rate")
        
        return {
            "problem": problem,
            "answer": answer,
            "source": source,
            "domain": domain,
            "llama8b_solve_rate": solve_rate,
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
            question = example["problem"] + "\n\n" + instruction
        else:
            if budget_window > 0:
                budget_str = TOKEN_BUDGET_WINDOW_STR.format(budget=budget, window=budget_window)
            else:
                budget_str = TOKEN_BUDGET_STR.format(budget=budget)
            question = example["problem"] + "\n\n" + budget_str + " " + instruction
        
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": example["answer"]},
            "extra_info": {
                "split": example["split"],
                "index": example["index"],
                "question": example["problem"],
                "answer": example["answer"],
                "source": example["source"],
                "domain": example["domain"],
                "llama8b_solve_rate": example["llama8b_solve_rate"],
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


def bigmath_token_budget(
    local_save_dir: str,
    num_budget_copies: int,
    budget_values: list,
    sources: Optional[list] = None,
    max_samples: Optional[int] = None,
    solve_rate_min: float = 0.0,
    solve_rate_max: float = 1.0,
    train_fraction: float = 0.8,
    numerical_only: bool = False,
    format_only_answer: bool = False,
    budget_window: int = 0,
    decreasing_budgets: bool = False,
    budget_range: bool = False,
    curriculum_warmup: Optional[float] = None,
    curriculum_cooldown: Optional[float] = None,
    seed: int = 42
):
    """
    Process Big-Math-RL-Verified dataset with token budgets.
    
    Args:
        local_save_dir: Directory to save the processed dataset
        num_budget_copies: Number of copies of each problem with different budgets
        budget_values: List of possible token budget values (can include 'control')
        sources: List of sources to include (None means all)
        max_samples: Maximum total number of samples (None means all)
        solve_rate_min: Minimum llama8b_solve_rate
        solve_rate_max: Maximum llama8b_solve_rate
        train_fraction: Fraction of data to use for training
        numerical_only: If True, only include problems with numerical answers
        format_only_answer: If True, instruct model to output only boxed answer with no explanation
        budget_window: Window around budget where no penalty is applied (default 0)
        decreasing_budgets: If True, order examples by budget (descending) instead of random sampling
        budget_range: If True, interpret budget_values as range endpoints
        curriculum_warmup: Fraction of dataset using initial budget (curriculum mode)
        curriculum_cooldown: Fraction of dataset using final budget (curriculum mode)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    data_source = "SynthLabsAI/Big-Math-RL-Verified"
    
    # Load dataset
    dataset = datasets.load_dataset(data_source, split="train")
    
    # Apply filters
    if sources is not None:
        dataset = dataset.filter(lambda x: x["source"] in sources)
    
    # Filter by solve rate range
    dataset = dataset.filter(
        lambda x: x["llama8b_solve_rate"] is not None and solve_rate_min <= x["llama8b_solve_rate"] <= solve_rate_max
    )
    
    # Filter for numerical answers if requested
    if numerical_only:
        dataset = dataset.filter(lambda x: is_numerical(x["answer"]))
    
    # Limit to max_samples if specified
    if max_samples is not None and len(dataset) > max_samples:
        # Shuffle and select
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
    
    # Process base data
    dataset = dataset.map(function=make_base_map_fn("base"), with_indices=True)
    
    # Split into train/test
    split_dataset = dataset.train_test_split(
        test_size=1.0 - train_fraction,
        seed=seed
    )
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    
    # Update split labels
    train_dataset = train_dataset.map(lambda x: {"split": "train"})
    test_dataset = test_dataset.map(lambda x: {"split": "test"})
    
    # Expand with different budgets
    train_dataset = expand_with_budgets(
        train_dataset, num_budget_copies, budget_values, data_source,
        format_only_answer, budget_window, decreasing_budgets,
        budget_range, curriculum_warmup, curriculum_cooldown,
        is_validation=False
    )
    test_dataset = expand_with_budgets(
        test_dataset, num_budget_copies, budget_values, data_source,
        format_only_answer, budget_window, decreasing_budgets,
        budget_range, curriculum_warmup, curriculum_cooldown,
        is_validation=True
    )
    
    # Save to parquet
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    
    print(f"Saved {len(train_dataset)} training examples to {os.path.join(local_save_dir, 'train.parquet')}")
    print(f"Saved {len(test_dataset)} test examples to {os.path.join(local_save_dir, 'test.parquet')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Big-Math-RL-Verified dataset with token budgets")
    
    parser.add_argument(
        "--local-save-dir",
        help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--num-budget-copies",
        type=int,
        default=1,
        help="Number of copies of each problem with different budgets."
    )
    parser.add_argument(
        "--budget-values",
        nargs='+',
        default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        help="Possible token budget values to sample from. Can include 'control' for no budget constraint."
    )
    parser.add_argument(
        "--sources",
        nargs='+',
        default=None,
        help="List of sources to include. If not specified, all sources are included."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total number of samples. If not specified, all samples are used."
    )
    parser.add_argument(
        "--solve-rate-min",
        type=float,
        default=0.0,
        help="Minimum llama8b_solve_rate (default: 0.0)"
    )
    parser.add_argument(
        "--solve-rate-max",
        type=float,
        default=1.0,
        help="Maximum llama8b_solve_rate (default: 1.0)"
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8)"
    )
    parser.add_argument(
        "--numerical-only",
        action="store_true",
        help="Only include problems with numerical answers"
    )
    parser.add_argument(
        "--format-only-answer",
        action="store_true",
        help="Instruct model to output only boxed answer with no explanation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
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
        if args.num_budget_copies > len(budget_values):
            raise ValueError(
                f"num_budget_copies ({args.num_budget_copies}) cannot exceed the number of "
                f"available budget values ({len(budget_values)})"
            )
        if args.decreasing_budgets and len(budget_values) < 2:
            raise ValueError("--decreasing-budgets requires at least 2 budget values")
    
    if (args.curriculum_warmup is not None or args.curriculum_cooldown is not None) and not args.decreasing_budgets:
        raise ValueError("--curriculum-warmup and --curriculum-cooldown require --decreasing-budgets")
    
    if args.curriculum_warmup is not None and args.curriculum_cooldown is not None:
        if args.curriculum_warmup + args.curriculum_cooldown > 1.0:
            raise ValueError("curriculum_warmup + curriculum_cooldown cannot exceed 1.0")
    
    bigmath_token_budget(
        local_save_dir=args.local_save_dir,
        num_budget_copies=args.num_budget_copies,
        budget_values=budget_values,
        sources=args.sources,
        max_samples=args.max_samples,
        solve_rate_min=args.solve_rate_min,
        solve_rate_max=args.solve_rate_max,
        train_fraction=args.train_fraction,
        numerical_only=args.numerical_only,
        format_only_answer=args.format_only_answer,
        budget_window=args.budget_window,
        decreasing_budgets=args.decreasing_budgets,
        budget_range=args.budget_range,
        curriculum_warmup=args.curriculum_warmup,
        curriculum_cooldown=args.curriculum_cooldown,
        seed=args.seed
    )

