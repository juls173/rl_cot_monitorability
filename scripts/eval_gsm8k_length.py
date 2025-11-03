import argparse
import time
import json
import os
from typing import Optional
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModelForCausalLM

# Import reward computation functions
from gsm8k_reward_length import compute_reward_breakdown


def load_gsm8k_parquet(path: str) -> pd.DataFrame:
    """Load GSM8K parquet file and extract relevant fields."""
    df = pd.read_parquet(path)
    
    # Extract fields from nested structure
    rows = []
    for _, row in df.iterrows():
        prompt_content = row['prompt'][0]['content']  # user message
        ground_truth = row['reward_model']['ground_truth']
        extra_info = row['extra_info']
        budget = extra_info['budget']
        
        rows.append({
            'prompt': prompt_content,
            'ground_truth': ground_truth,
            'budget': budget,
            'question': extra_info['question'],
            'answer': extra_info['answer'],
        })
    
    return pd.DataFrame(rows)


def save_jsonl(path: str, data: list):
    """Save list of dicts to JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


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
    length_reward: Optional[float] = None,
    length_exponent: Optional[float] = None,
):
    """Run evaluation on GSM8K with length constraints.
    
    If lora_path is provided, model should be the base model and lora_path
    should point to the LoRA adapter directory.
    
    If length_reward/length_exponent are provided, they will be used for reward
    computation; otherwise they are loaded from environment variables.
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
    
    # Prepare prompts
    prompts = df["prompt"].tolist()
    
    # Run inference
    outputs = pipe(
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=n_samples,
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=False,
        batch_size=batch_size,
    )
    
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
    length_compliant_count = 0
    both_correct_count = 0
    total_reward_sum = 0.0
    
    for i, texts in enumerate(all_generated_texts):
        ground_truth = df.iloc[i]["ground_truth"]
        budget = int(df.iloc[i]["budget"])
        prompt = df.iloc[i]["prompt"]
        
        generated_text = texts[0]  # Pass@1 = first sample
        
        # Use compute_reward_breakdown from gsm8k_reward_length.py
        metrics = compute_reward_breakdown(
            generated_text, 
            ground_truth, 
            budget, 
            length_reward=length_reward,
            length_exponent=length_exponent,
            tokenizer=tokenizer
        )
        
        is_correct = metrics["correctness_reward"] > 0.5  # Convert to boolean
        is_length_ok = metrics["length_diff"] <= 50
        
        correct_count += int(is_correct)
        length_compliant_count += int(is_length_ok)
        both_correct_count += int(is_correct and is_length_ok)
        total_reward_sum += metrics["total_reward"]
        
        rows.append({
            "idx": i,
            "question": df.iloc[i]["question"],
            "budget": budget,
            "ground_truth": ground_truth,
            "prompt": prompt,
            "generated_text": generated_text,
            "predicted_answer": metrics["extracted_answer"],
            "token_count": metrics["token_count"],
            "length_diff": metrics["length_diff"],
            "correctness_reward": metrics["correctness_reward"],
            "length_bonus": metrics["length_bonus"],
            "total_reward": metrics["total_reward"],
            "is_correct": is_correct,
            "is_length_compliant": is_length_ok,
            "is_both_ok": is_correct and is_length_ok,
            "all_texts": texts if n_samples > 1 else None,
        })
    
    # Print summary statistics
    n = len(df)
    acc = correct_count / n
    length_acc = length_compliant_count / n
    both_acc = both_correct_count / n
    avg_reward = total_reward_sum / n
    
    print(f"Evaluation Results:")
    print(f"  N={n}")
    print(f"  Correctness: {correct_count}/{n} = {acc:.4f}")
    print(f"  Length Compliance: {length_compliant_count}/{n} = {length_acc:.4f}")
    print(f"  Both Correct: {both_correct_count}/{n} = {both_acc:.4f}")
    print(f"  Average Reward: {avg_reward:.4f}")
    
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
    ap.add_argument("--length-reward", type=float, default=None, 
                    help="Reward for length compliance (loads from env if not provided)")
    ap.add_argument("--length-exponent", type=float, default=None,
                    help="Exponent for length penalty (loads from env if not provided)")
    
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
        length_reward=args.length_reward,
        length_exponent=args.length_exponent,
    )

