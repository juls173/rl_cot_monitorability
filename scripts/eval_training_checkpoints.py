import argparse
import os
import shutil
from typing import List, Optional

from eval_length import eval_length
from eval_readability_length import eval_readability_length


def eval_training_checkpoints(
    training_dir: str,
    steps: List[int],
    output_dir: str,
    model: str,
    data: str,
    # eval_length.py arguments
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
    max_model_len: int = 2048,
    n_samples: int = 1,
    batch_size: int = 8,
    limit: Optional[int] = None,
    length_penalty: Optional[float] = None,
    format_penalty: Optional[float] = None,
    forbidden_word: Optional[str] = None,
    # eval_readability_length.py arguments
    run_readability: bool = False,
    openrouter_api_key: Optional[str] = None,
    max_concurrent: int = 10,
    readability_plot: bool = False,
    extract_cot: bool = False,
    readability_model: str = "x-ai/grok-4-fast",
):
    """Run eval_length.py and optionally eval_readability_length.py on training checkpoints.
    
    Args:
        training_dir: Directory containing training checkpoints (global_step_###/actor/lora_adapter)
        steps: List of step numbers to evaluate
        output_dir: Directory to save results
        model: Base model path for eval_length.py
        data: Dataset path for eval_length.py
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_new_tokens: Maximum new tokens to generate
        max_model_len: Maximum model context length
        n_samples: Number of samples per prompt
        batch_size: Batch size for inference
        limit: Limit number of samples to evaluate
        length_penalty: Length penalty coefficient
        format_penalty: Format penalty coefficient
        forbidden_word: Forbidden word to check in reasoning
        run_readability: Whether to run readability evaluation
        openrouter_api_key: API key for OpenRouter (for readability eval)
        max_concurrent: Max concurrent requests for readability eval
        readability_plot: Whether to generate readability plots
        extract_cot: Whether to extract only CoT for readability eval
        readability_model: Model to use for readability evaluation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for step in steps:
        print(f"\n{'='*60}")
        print(f"Processing step {step}")
        print(f"{'='*60}")
        
        # Source LoRA path
        lora_src = os.path.join(training_dir, f"global_step_{step}", "actor", "lora_adapter")
        if not os.path.exists(lora_src):
            raise FileNotFoundError(f"LoRA adapter not found at {lora_src}")
        
        # Create step output directory
        step_dir = os.path.join(output_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Copy LoRA to output directory
        lora_dst = os.path.join(step_dir, "lora_adapter")
        if os.path.exists(lora_dst):
            shutil.rmtree(lora_dst)
        shutil.copytree(lora_src, lora_dst)
        print(f"Copied LoRA from {lora_src} to {lora_dst}")
        
        # Run eval_length
        eval_length_out = os.path.join(step_dir, "eval_length.jsonl")
        print(f"\nRunning eval_length for step {step}...")
        eval_length(
            model=model,
            data=data,
            out=eval_length_out,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
            n_samples=n_samples,
            batch_size=batch_size,
            limit=limit,
            lora_path=lora_dst,
            length_penalty=length_penalty,
            format_penalty=format_penalty,
            forbidden_word=forbidden_word,
        )
        print(f"Saved eval_length results to {eval_length_out}")
        
        # Optionally run eval_readability_length
        if run_readability:
            readability_out = os.path.join(step_dir, "eval_readability.json")
            plot_out = os.path.join(step_dir, "readability_plot.png") if readability_plot else None
            
            print(f"\nRunning eval_readability_length for step {step}...")
            eval_readability_length(
                input_jsonl=eval_length_out,
                output_json=readability_out,
                openrouter_api_key=openrouter_api_key,
                max_concurrent=max_concurrent,
                plot_output=plot_out,
                extract_cot=extract_cot,
                model=readability_model,
            )
            print(f"Saved readability results to {readability_out}")
    
    print(f"\n{'='*60}")
    print(f"Completed evaluation for all {len(steps)} steps")
    print(f"Results saved to {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate training checkpoints with eval_length.py and optionally eval_readability_length.py"
    )
    
    # Required arguments
    parser.add_argument(
        "--training-dir",
        required=True,
        help="Training directory containing global_step_###/actor/lora_adapter subdirectories"
    )
    parser.add_argument(
        "--steps",
        required=True,
        type=str,
        help="Comma-separated list of step numbers to evaluate (e.g., '100,200,300')"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory to save results and copy LoRAs"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to base model for eval_length.py"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to parquet dataset for eval_length.py"
    )
    
    # eval_length.py arguments
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--length-penalty", type=float, default=None)
    parser.add_argument("--format-penalty", type=float, default=None)
    parser.add_argument("--forbidden-word", type=str, default=None)
    
    # eval_readability_length.py arguments
    parser.add_argument(
        "--run-readability",
        action="store_true",
        help="Run eval_readability_length.py on the results"
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=None,
        help="OpenRouter API key for readability evaluation"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent requests for readability evaluation"
    )
    parser.add_argument(
        "--readability-plot",
        action="store_true",
        help="Generate readability plots"
    )
    parser.add_argument(
        "--extract-cot",
        action="store_true",
        help="Extract only CoT for readability evaluation"
    )
    parser.add_argument(
        "--readability-model",
        default="x-ai/grok-4-fast",
        help="Model to use for readability evaluation"
    )
    
    args = parser.parse_args()
    
    # Parse steps from comma-separated string
    steps = [int(s.strip()) for s in args.steps.split(",")]
    
    eval_training_checkpoints(
        training_dir=args.training_dir,
        steps=steps,
        output_dir=args.output_dir,
        model=args.model,
        data=args.data,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        limit=args.limit,
        length_penalty=args.length_penalty,
        format_penalty=args.format_penalty,
        forbidden_word=args.forbidden_word,
        run_readability=args.run_readability,
        openrouter_api_key=args.openrouter_api_key,
        max_concurrent=args.max_concurrent,
        readability_plot=args.readability_plot,
        extract_cot=args.extract_cot,
        readability_model=args.readability_model,
    )

