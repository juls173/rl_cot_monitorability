import argparse
import time
import json
import re
import os
from typing import Optional, Literal
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

# Export VLLM configuration
# os.environ["VLLM_USE_V1"] = "0"

def extract_answer(solution_str: str, method: Literal["strict", "flexible"] = "flexible") -> Optional[str]:
    """Extract numerical answer from various formats."""
    
    # Method 1: Try LaTeX \boxed{} format (DeepSeek R1 style)
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution_str)
    if boxed_match:
        answer = boxed_match.group(1).strip()
        answer = answer.replace(',', '').replace('$', '').strip()
        return answer
    
    # Method 2: Try GSM8K #### format
    gsm8k_match = re.search(r'####\s*(\-?[\d\.\,]+)', solution_str)
    if gsm8k_match:
        answer = gsm8k_match.group(1).replace(',', '').replace('$', '').strip()
        return answer
    
    # Method 3: Flexible - find last number in text
    if method == "flexible":
        numbers = re.findall(r'\-?[\d\.\,]+', solution_str)
        if numbers:
            return numbers[-1].replace(',', '').strip()
    
    return None


def compute_metrics(solution_str: str, ground_truth: str, budget: int, tokenizer) -> dict:
    """Compute correctness, length compliance, and combined reward."""
    # Compute correctness
    predicted_answer = extract_answer(solution_str, method="flexible")
    if predicted_answer is None:
        correctness = False
    else:
        gt_clean = str(ground_truth).replace(',', '').strip()
        
        try:
            pred_num = float(predicted_answer)
            gt_num = float(gt_clean)
            correctness = abs(pred_num - gt_num) < 1e-6
        except ValueError:
            correctness = predicted_answer == gt_clean
    
    # Compute length metrics
    token_count = len(tokenizer.encode(solution_str))
    length_diff = abs(token_count - budget)
    length_compliant = length_diff <= 50
    
    return {
        "correctness": correctness,
        "token_count": token_count,
        "budget": budget,
        "length_diff": length_diff,
        "length_compliant": length_compliant,
        "predicted_answer": predicted_answer,
    }


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
    limit: Optional[int] = None,
    lora_path: Optional[str] = None,
):
    """Run evaluation on GSM8K with length constraints.
    
    If lora_path is provided, model should be the base model and lora_path
    should point to the LoRA adapter directory.
    """
    # Load data
    df = load_gsm8k_parquet(data)
    if limit:
        df = df.head(limit)
    
    # Load tokenizer for length computation
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # Prepare sampling params
    sp = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=n_samples,
    )
    
    # Run inference
    if lora_path:        
        llm = LLM(model=model, enable_lora=True, max_lora_rank=32, max_model_len=max_model_len)
        lora_request = LoRARequest("adapter", 1, lora_path)
        prompts = df["prompt"].tolist()
        t0 = time.time()
        outputs = llm.generate(prompts, sp, lora_request=lora_request)
        dt = time.time() - t0
    else:
        llm = LLM(model=model, max_model_len=max_model_len)
        prompts = df["prompt"].tolist()
        t0 = time.time()
        outputs = llm.generate(prompts, sp)
        dt = time.time() - t0
    
    # Evaluate results
    rows = []
    correct_count = 0
    length_compliant_count = 0
    both_correct_count = 0
    
    for i, output in enumerate(outputs):
        ground_truth = df.iloc[i]["ground_truth"]
        budget = int(df.iloc[i]["budget"])
        
        texts = [c.text for c in output.outputs]  # n samples per prompt
        chosen = texts[0]  # Pass@1 = first sample
        
        metrics = compute_metrics(chosen, ground_truth, budget, tokenizer)
        
        is_correct = metrics["correctness"]
        is_length_ok = metrics["length_compliant"]
        
        correct_count += int(is_correct)
        length_compliant_count += int(is_length_ok)
        both_correct_count += int(is_correct and is_length_ok)
        
        rows.append({
            "idx": i,
            "question": df.iloc[i]["question"],
            "budget": budget,
            "ground_truth": ground_truth,
            "chosen_text": chosen,
            "predicted_answer": metrics["predicted_answer"],
            "token_count": metrics["token_count"],
            "length_diff": metrics["length_diff"],
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
    
    print(f"Evaluation Results:")
    print(f"  N={n}")
    print(f"  Correctness: {correct_count}/{n} = {acc:.4f}")
    print(f"  Length Compliance: {length_compliant_count}/{n} = {length_acc:.4f}")
    print(f"  Both Correct: {both_correct_count}/{n} = {both_acc:.4f}")
    print(f"  Time: {dt:.1f}s")
    
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
    ap.add_argument("--limit", type=int, default=None)
    
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
        limit=args.limit,
        lora_path=args.lora_path,
    )

