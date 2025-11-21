"""
Preprocess the Big-Math-RL-Verified dataset to parquet format with token budgets
"""

import argparse
import os
import random

import datasets


INSTRUCTION_FOLLOWING = "Think step by step and output the final answer using \\boxed{}."
INSTRUCTION_FOLLOWING_FORMAT_ONLY = "Think step by step and output ONLY the final answer using \\boxed{}. Do not provide any other explanation."
TOKEN_BUDGET_STR = 'You have a token budget of around {budget} tokens. You must finish your thinking process as close as possible to the thinking budget.'


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


def expand_with_budgets(dataset, num_copies, budget_values, data_source, format_only_answer=False):
    """Create multiple copies of each example with different token budgets."""
    expanded = []
    
    instruction = INSTRUCTION_FOLLOWING_FORMAT_ONLY if format_only_answer else INSTRUCTION_FOLLOWING
    
    for example in dataset:
        budgets = random.sample(budget_values, num_copies)
        
        for copy_idx, budget in enumerate(budgets):
            # Handle control condition (no budget constraint)
            if budget == 'control':
                question = example["problem"] + "\n\n" + instruction
            else:
                question = (
                    example["problem"] + "\n\n" + 
                    TOKEN_BUDGET_STR.format(budget=budget) + " " + 
                    instruction
                )
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
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
                    "copy_idx": copy_idx,
                    "format_only_answer": format_only_answer,
                },
            }
            expanded.append(data)
    
    return datasets.Dataset.from_list(expanded)


def bigmath_token_budget(
    local_save_dir,
    num_budget_copies,
    budget_values,
    sources=None,
    max_samples=None,
    solve_rate_min=0.0,
    solve_rate_max=1.0,
    train_fraction=0.8,
    numerical_only=False,
    format_only_answer=False,
    seed=42
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
    train_dataset = expand_with_budgets(train_dataset, num_budget_copies, budget_values, data_source, format_only_answer)
    test_dataset = expand_with_budgets(test_dataset, num_budget_copies, budget_values, data_source, format_only_answer)
    
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
    
    args = parser.parse_args()
    
    # Convert budget values to int except for 'control'
    budget_values = []
    for val in args.budget_values:
        if val == 'control':
            budget_values.append('control')
        else:
            budget_values.append(int(val))
    
    if args.num_budget_copies > len(budget_values):
        raise ValueError(
            f"num_budget_copies ({args.num_budget_copies}) cannot exceed the number of "
            f"available budget values ({len(budget_values)})"
        )
    
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
        seed=args.seed
    )

