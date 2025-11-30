import argparse
import json
from typing import Optional
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModelForCausalLM
from tqdm import tqdm

# Import reward computation functions
from reward_length import compute_reward_breakdown


def load_gsm8k_parquet(path: str) -> pd.DataFrame:
    """Load GSM8K parquet file and extract relevant fields."""
    df = pd.read_parquet(path)
    
    # Extract fields from nested structure
    rows = []
    for _, row in df.iterrows():
        prompt_messages = row['prompt']  # Keep full message list for chat template
        ground_truth = row['reward_model']['ground_truth']
        extra_info = row['extra_info']
        budget = extra_info['budget']
        budget_window = extra_info.get('budget_window', 0)
        
        rows.append({
            'prompt': prompt_messages,
            'ground_truth': ground_truth,
            'budget': budget,
            'budget_window': budget_window,
            'question': extra_info['question'],
            'answer': extra_info['answer'],
            'format_only_answer': extra_info.get('format_only_answer', False),
        })
    
    return pd.DataFrame(rows)


def save_jsonl(path: str, data: list):
    """Save list of dicts to JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def count_forbidden_word_in_thinking(text: str, forbidden_word: str) -> int:
    """Count occurrences of forbidden word before the closing </think> tag.
    
    Args:
        text: Generated text that may contain <think>...</think> tags
        forbidden_word: The word to count (case-insensitive)
    
    Returns:
        Number of times the forbidden word appears before </think>
    """
    # Find the closing </think> tag
    think_end_idx = text.find('</think>')
    if think_end_idx == -1:
        # No closing think tag found, search the entire text
        search_text = text
    else:
        # Only search before the closing tag
        search_text = text[:think_end_idx]
    
    # Count occurrences (case-insensitive)
    return search_text.lower().count(forbidden_word.lower())


def eval_gsm8k_length(
    model: str,
    data: str,
    out: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
    max_model_len: int = 2048,
    n_samples: int = 1,
    batch_size: int = 8,
    limit: Optional[int] = None,
    lora_path: Optional[str] = None,
    length_penalty: Optional[float] = None,
    format_penalty: Optional[float] = None,
    forbidden_word: Optional[str] = None,
):
    """Run evaluation on GSM8K with length constraints.
    
    If lora_path is provided, model should be the base model and lora_path
    should point to the LoRA adapter directory.
    
    If length_penalty is provided, it will be used for reward computation;
    otherwise it is loaded from environment variables.
    
    If format_penalty is provided, it will be used for reward computation;
    otherwise it is loaded from environment variables.
    """
    # Load data
    df = load_gsm8k_parquet(data)
    if limit:
        df = df.head(limit)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    llm = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    
    # Load LoRA adapter if provided
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path}...")
        llm = PeftModelForCausalLM.from_pretrained(llm, lora_path)
    
    llm.eval()
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    # Prepare prompts - apply chat template to message lists
    prompt_messages = df["prompt"].tolist()
    
    # Add forbidden word instruction if provided
    if forbidden_word:
        forbidden_instruction = f'Do NOT use the word "{forbidden_word}" in your private reasoning. You MUST NOT use the word "{forbidden_word}" in your private reasoning.'
        # Append instruction to the last user message in each prompt
        modified_prompt_messages = []
        for messages in prompt_messages:
            modified_messages = messages.copy()
            # Find the last user message and append the instruction
            for i in range(len(modified_messages) - 1, -1, -1):
                if modified_messages[i].get('role') == 'user':
                    modified_messages[i] = modified_messages[i].copy()
                    modified_messages[i]['content'] = modified_messages[i]['content'] + '\n\n' + forbidden_instruction
                    break
            modified_prompt_messages.append(modified_messages)
        prompt_messages = modified_prompt_messages
    
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in prompt_messages
    ]
    
    # Run inference
    outputs = list(tqdm(
        pipe(
            (x for x in prompts),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=n_samples,
            pad_token_id=tokenizer.pad_token_id,
            return_full_text=False,
            batch_size=batch_size,
        ),
        total=len(prompts),
        desc="Generating responses"
    ))
    
    # Extract generated texts
    all_generated_texts = []
    for output in outputs:
        if n_samples == 1:
            # Single sample case: output is a list with one dict
            texts = [output[0]["generated_text"]]
        else:
            # Multiple samples: output is a list of dicts
            texts = [item["generated_text"] for item in output]
        all_generated_texts.append(texts)
        
    # Evaluate results
    rows = []
    correct_count = 0
    total_reward_sum = 0.0
    total_distance_sum = 0.0
    distance_count = 0  # Count of non-control samples
    total_format_penalty_sum = 0.0
    forbidden_word_violation_count = 0
    total_forbidden_word_count = 0
    
    for i, texts in enumerate(all_generated_texts):
        ground_truth = df.iloc[i]["ground_truth"]
        budget = df.iloc[i]["budget"]
        # Convert budget to int only if it's not 'control'
        if budget != 'control':
            budget = int(budget)
        budget_window = int(df.iloc[i]["budget_window"])
        prompt_messages = df.iloc[i]["prompt"]
        formatted_prompt = prompts[i]
        
        generated_text = texts[0]  # Pass@1 = first sample
        
        # Use compute_reward_breakdown from reward_length.py
        metrics = compute_reward_breakdown(
            generated_text, 
            ground_truth, 
            budget, 
            budget_window=budget_window,
            length_penalty=length_penalty,
            format_penalty=format_penalty,
            tokenizer=tokenizer
        )
        
        is_correct = metrics["correctness_reward"] > 0.5  # Convert to boolean
        
        # Track distance from budget (for non-control samples)
        if metrics["length_diff"] is not None:
            total_distance_sum += abs(metrics["length_diff"])
            distance_count += 1
        
        # Count forbidden word occurrences if specified
        forbidden_word_count = None
        has_forbidden_word_violation = None
        if forbidden_word:
            forbidden_word_count = count_forbidden_word_in_thinking(generated_text, forbidden_word)
            has_forbidden_word_violation = forbidden_word_count > 0
            forbidden_word_violation_count += int(has_forbidden_word_violation)
            total_forbidden_word_count += forbidden_word_count
        
        correct_count += int(is_correct)
        total_reward_sum += metrics["total_reward"]
        total_format_penalty_sum += metrics["format_penalty"]
        
        rows.append({
            "idx": i,
            "question": df.iloc[i]["question"],
            "budget": budget,
            "budget_window": budget_window,
            "ground_truth": ground_truth,
            "format_only_answer": df.iloc[i]["format_only_answer"],
            "prompt": formatted_prompt,
            "generated_text": generated_text,
            "predicted_answer": metrics["extracted_answer"],
            "token_count": metrics["token_count"],
            "length_diff": metrics["length_diff"],
            "correctness_reward": metrics["correctness_reward"],
            "length_penalty": metrics["length_penalty"],
            "format_penalty": metrics["format_penalty"],
            "total_reward": metrics["total_reward"],
            "is_correct": is_correct,
            "forbidden_word": forbidden_word,
            "forbidden_word_count": forbidden_word_count,
            "has_forbidden_word_violation": has_forbidden_word_violation,
            "all_texts": texts if n_samples > 1 else None,
        })
    
    # Print summary statistics
    n = len(df)
    acc = correct_count / n
    avg_reward = total_reward_sum / n
    avg_format_penalty = total_format_penalty_sum / n
    
    # Compute overall distance metrics
    results_df_temp = pd.DataFrame(rows)
    non_null_df = results_df_temp[results_df_temp['length_diff'].notna()]
    
    avg_distance = None
    avg_distance_correct = None
    avg_distance_incorrect = None
    
    if len(non_null_df) > 0:
        avg_distance = non_null_df['length_diff'].abs().mean()
        
        correct_df = non_null_df[non_null_df['is_correct'] == True]
        incorrect_df = non_null_df[non_null_df['is_correct'] == False]
        
        avg_distance_correct = correct_df['length_diff'].abs().mean() if len(correct_df) > 0 else None
        avg_distance_incorrect = incorrect_df['length_diff'].abs().mean() if len(incorrect_df) > 0 else None
    
    print(f"Overall Evaluation Results:")
    print(f"- N={n}")
    print(f"- Correctness: {correct_count}/{n} = {acc:.4f}")
    print(f"- Average Reward: {avg_reward:.4f}")
    print(f"- Average Format Penalty: {avg_format_penalty:.4f}")
    if avg_distance is not None:
        print(f"- Average Distance from Budget: {avg_distance:.2f} tokens")
        if avg_distance_correct is not None:
            print(f"- Average Distance (Correct): {avg_distance_correct:.2f} tokens")
        if avg_distance_incorrect is not None:
            print(f"- Average Distance (Incorrect): {avg_distance_incorrect:.2f} tokens")
    if forbidden_word:
        violation_rate = forbidden_word_violation_count / n
        avg_forbidden_count = total_forbidden_word_count / n
        print(f"- Forbidden Word '{forbidden_word}' Violations: {forbidden_word_violation_count}/{n} = {violation_rate:.4f}")
        print(f"- Avg Forbidden Word Count: {avg_forbidden_count:.4f}")
    print()
    
    # Compute per-budget statistics
    results_df = pd.DataFrame(rows)
    
    if forbidden_word:
        budget_stats = results_df.groupby('budget').agg({
            'is_correct': ['sum', 'count', 'mean'],
            'total_reward': 'mean',
            'format_penalty': 'mean',
            'has_forbidden_word_violation': 'mean',
            'forbidden_word_count': 'mean'
        }).reset_index()
        
        # Flatten column names
        budget_stats.columns = ['budget', 'correct_count', 'total_count', 'accuracy', 
                                'avg_reward', 'avg_format_penalty', 'violation_rate', 'avg_forbidden_count']
        
        # Compute average distance metrics per budget
        # For non-control samples, compute average distance and conditional distances
        for budget in budget_stats['budget'].unique():
            budget_df = results_df[results_df['budget'] == budget]
            non_null_df = budget_df[budget_df['length_diff'].notna()]
            
            if len(non_null_df) > 0:
                avg_dist = non_null_df['length_diff'].abs().mean()
                
                correct_df = non_null_df[non_null_df['is_correct'] == True]
                incorrect_df = non_null_df[non_null_df['is_correct'] == False]
                
                avg_dist_correct = correct_df['length_diff'].abs().mean() if len(correct_df) > 0 else None
                avg_dist_incorrect = incorrect_df['length_diff'].abs().mean() if len(incorrect_df) > 0 else None
                
                budget_stats.loc[budget_stats['budget'] == budget, 'avg_distance'] = avg_dist
                budget_stats.loc[budget_stats['budget'] == budget, 'avg_distance_correct'] = avg_dist_correct
                budget_stats.loc[budget_stats['budget'] == budget, 'avg_distance_incorrect'] = avg_dist_incorrect
            else:
                budget_stats.loc[budget_stats['budget'] == budget, 'avg_distance'] = None
                budget_stats.loc[budget_stats['budget'] == budget, 'avg_distance_correct'] = None
                budget_stats.loc[budget_stats['budget'] == budget, 'avg_distance_incorrect'] = None
        
        # Sort by budget, putting 'control' at the end
        budget_stats['sort_key'] = budget_stats['budget'].apply(
            lambda x: float('inf') if x == 'control' else x
        )
        budget_stats = budget_stats.sort_values('sort_key').drop(columns=['sort_key'])
        
        print("Per-Budget Results:")
        print("| Budget | N | Accuracy | Avg Reward | Avg Format Penalty | Avg Distance | Avg Dist (Correct) | Avg Dist (Incorrect) | Violation Rate | Avg Forbidden Count |")
        print("|--------|---|----------|------------|--------------------|--------------|--------------------|----------------------|----------------|---------------------|")
        for _, row in budget_stats.iterrows():
            budget_str = str(row['budget']) if row['budget'] == 'control' else str(int(row['budget']))
            dist_str = f"{row['avg_distance']:.2f}" if pd.notna(row['avg_distance']) else "N/A"
            dist_correct_str = f"{row['avg_distance_correct']:.2f}" if pd.notna(row['avg_distance_correct']) else "N/A"
            dist_incorrect_str = f"{row['avg_distance_incorrect']:.2f}" if pd.notna(row['avg_distance_incorrect']) else "N/A"
            print(f"| {budget_str} | {int(row['total_count'])} | {row['accuracy']:.4f} | "
                  f"{row['avg_reward']:.4f} | {row['avg_format_penalty']:.4f} | {dist_str} | {dist_correct_str} | {dist_incorrect_str} | "
                  f"{row['violation_rate']:.4f} | {row['avg_forbidden_count']:.4f} |")
        print()
    else:
        budget_stats = results_df.groupby('budget').agg({
            'is_correct': ['sum', 'count', 'mean'],
            'total_reward': 'mean',
            'format_penalty': 'mean'
        }).reset_index()
        
        # Flatten column names
        budget_stats.columns = ['budget', 'correct_count', 'total_count', 'accuracy', 'avg_reward', 'avg_format_penalty']
        
        # Compute average distance metrics per budget
        # For non-control samples, compute average distance and conditional distances
        for budget in budget_stats['budget'].unique():
            budget_df = results_df[results_df['budget'] == budget]
            non_null_df = budget_df[budget_df['length_diff'].notna()]
            
            if len(non_null_df) > 0:
                avg_dist = non_null_df['length_diff'].abs().mean()
                
                correct_df = non_null_df[non_null_df['is_correct'] == True]
                incorrect_df = non_null_df[non_null_df['is_correct'] == False]
                
                avg_dist_correct = correct_df['length_diff'].abs().mean() if len(correct_df) > 0 else None
                avg_dist_incorrect = incorrect_df['length_diff'].abs().mean() if len(incorrect_df) > 0 else None
                
                budget_stats.loc[budget_stats['budget'] == budget, 'avg_distance'] = avg_dist
                budget_stats.loc[budget_stats['budget'] == budget, 'avg_distance_correct'] = avg_dist_correct
                budget_stats.loc[budget_stats['budget'] == budget, 'avg_distance_incorrect'] = avg_dist_incorrect
            else:
                budget_stats.loc[budget_stats['budget'] == budget, 'avg_distance'] = None
                budget_stats.loc[budget_stats['budget'] == budget, 'avg_distance_correct'] = None
                budget_stats.loc[budget_stats['budget'] == budget, 'avg_distance_incorrect'] = None
        
        # Sort by budget, putting 'control' at the end
        budget_stats['sort_key'] = budget_stats['budget'].apply(
            lambda x: float('inf') if x == 'control' else x
        )
        budget_stats = budget_stats.sort_values('sort_key').drop(columns=['sort_key'])
        
        print("Per-Budget Results:")
        print("| Budget | N | Accuracy | Avg Reward | Avg Format Penalty | Avg Distance | Avg Dist (Correct) | Avg Dist (Incorrect) |")
        print("|--------|---|----------|------------|--------------------|--------------|--------------------|----------------------|")
        for _, row in budget_stats.iterrows():
            budget_str = str(row['budget']) if row['budget'] == 'control' else str(int(row['budget']))
            dist_str = f"{row['avg_distance']:.2f}" if pd.notna(row['avg_distance']) else "N/A"
            dist_correct_str = f"{row['avg_distance_correct']:.2f}" if pd.notna(row['avg_distance_correct']) else "N/A"
            dist_incorrect_str = f"{row['avg_distance_incorrect']:.2f}" if pd.notna(row['avg_distance_incorrect']) else "N/A"
            print(f"| {budget_str} | {int(row['total_count'])} | {row['accuracy']:.4f} | "
                  f"{row['avg_reward']:.4f} | {row['avg_format_penalty']:.4f} | {dist_str} | {dist_correct_str} | {dist_incorrect_str} |")
        print()
    
    # Compute correlation between budget and accuracy (excluding control samples)
    non_control_df = results_df[results_df['budget'] != 'control']
    if len(non_control_df) > 1:
        budgets = non_control_df['budget'].values
        accuracies = non_control_df['is_correct'].astype(int).values
        correlation = np.corrcoef(budgets, accuracies)[0, 1]
        print(f"Correlation between budget and accuracy (excluding control): {correlation:.4f}")
    else:
        print("Not enough non-control samples to compute correlation")
    print()
    
    # Save results
    save_jsonl(out, rows)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to base model (or full model if not using LoRA)")
    ap.add_argument("--data", required=True, help="Path to parquet dataset")
    ap.add_argument("--out", required=True, help="Output JSONL file")
    ap.add_argument("--lora-path", default=None, help="Path to LoRA adapter directory (optional)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--length-penalty", type=float, default=None, 
                    help="Penalty per token of distance from budget (loads from env if not provided)")
    ap.add_argument("--format-penalty", type=float, default=None,
                    help="Penalty for incorrect format (loads from env if not provided)")
    ap.add_argument("--forbidden-word", type=str, default=None,
                    help="Word to forbid in the model's private reasoning")
    
    args = ap.parse_args()
    
    eval_gsm8k_length(
        model=args.model,
        data=args.data,
        out=args.out,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        limit=args.limit,
        lora_path=args.lora_path,
        length_penalty=args.length_penalty,
        format_penalty=args.format_penalty,
        forbidden_word=args.forbidden_word,
    )

