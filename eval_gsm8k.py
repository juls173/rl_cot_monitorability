import argparse, time
from vllm import LLM, SamplingParams
from utils_gsm8k import load_gsm8k_parquet, extract_final_number, save_jsonl

PROMPT_TMPL = """Solve the following problem. Show your work, then give the final answer on a line that starts with ####.

{question}
"""

def run_eval(model, parquet, out, do_sample, temperature, top_p, max_new_tokens, n_samples=1, limit=None):
    df = load_gsm8k_parquet(parquet)
    if limit: df = df.head(limit)
    prompts = [PROMPT_TMPL.format(question=q) for q in df["question"].tolist()]

    sp = SamplingParams(
        temperature=float(temperature) if do_sample else 0.0,
        top_p=float(top_p) if do_sample else 1.0,
        max_tokens=int(max_new_tokens),
        n=n_samples,
    )

    llm = LLM(model=model)
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    dt = time.time() - t0

    rows, correct = [], 0
    for i, out in enumerate(outputs):
        gold_final = extract_final_number(df.iloc[i]["answer"])
        texts = [c.text for c in out.outputs]   # n samples per prompt
        chosen = texts[0]                       # Pass@1 = first sample
        pred_final = extract_final_number(chosen)
        is_ok = (pred_final is not None) and (gold_final is not None) and (pred_final == gold_final)
        correct += int(is_ok)
        rows.append({
            "idx": int(i),
            "question": df.iloc[i]["question"],
            "gold_final": gold_final,
            "chosen_text": chosen,
            "pred_final": pred_final,
            "is_correct": is_ok,
            "all_texts": texts if n_samples > 1 else None,
        })

    acc = correct / len(df)
    print(f"Eval: N={len(df)}  Acc={acc:.4f}  Time={dt:.1f}s")
    save_jsonl(out, rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--do-sample", type=lambda s: s.lower()=="true", default=False)
    ap.add_argument("--temperature", type=float, default=1.3)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run_eval(args.model, args.parquet, args.out, args.do_sample, args.temperature,
             args.top_p, args.max_new_tokens, args.n_samples, args.limit)
