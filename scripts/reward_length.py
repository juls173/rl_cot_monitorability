import os
import re
from typing import Literal, Optional
from transformers import AutoTokenizer

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Global variables (lazy-loaded from environment if not passed as arguments)
_LENGTH_PENALTY = None
_MODEL_NAME = None
_TOKENIZER = None
_FORMAT_PENALTY = None
_WARMUP_PERIOD = None
_CALL_COUNT = 0

# Accumulator for WandB logging (tracks metrics between split changes)
_CURRENT_SPLIT = None
_ACCUMULATED_METRICS: dict[str, list[float]] = {}

def _load_from_env():
    """Lazy-load configuration from environment variables."""
    global _LENGTH_PENALTY, _MODEL_NAME, _TOKENIZER, _FORMAT_PENALTY, _WARMUP_PERIOD
    if _LENGTH_PENALTY is None:
        _LENGTH_PENALTY = float(os.environ["LENGTH_PENALTY"])
    if _MODEL_NAME is None:
        _MODEL_NAME = os.environ["TOKENIZER_MODEL_NAME"]
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)
    if _FORMAT_PENALTY is None:
        _FORMAT_PENALTY = float(os.environ.get("FORMAT_ONLY_ANSWER_PENALTY", 0.0))
    if _WARMUP_PERIOD is None:
        warmup_str = os.environ.get("PENALTY_WARMUP_PERIOD")
        _WARMUP_PERIOD = int(warmup_str) if warmup_str is not None else 0


def _get_warmup_factor() -> float:
    """Get the current warmup factor based on call count and warmup period."""
    if _WARMUP_PERIOD is None or _WARMUP_PERIOD <= 0:
        return 1.0
    return min(1.0, _CALL_COUNT / _WARMUP_PERIOD)


def compute_length_penalty(token_count: int, budget: int, budget_window: int, length_penalty: float) -> float:
    """Compute the length penalty based on token count and budget window.
    
    The penalty is zero within budget ± budget_window, then increases linearly
    from the edge of the window.
    
    Args:
        token_count: Number of tokens in the solution
        budget: Target token budget
        budget_window: Window around budget where no penalty is applied
        length_penalty: Penalty per token outside the window
    
    Returns:
        Negative penalty value (or 0.0 if within window)
    """
    distance_from_budget = abs(token_count - budget)
    if distance_from_budget <= budget_window:
        return 0.0
    else:
        # Penalty starts from the edge of the window
        effective_distance = distance_from_budget - budget_window
        return -length_penalty * effective_distance

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

def compute_reward_breakdown(
    solution_str: str,
    ground_truth: str,
    budget,  # Can be int or 'control'
    budget_window: int = 0,
    length_penalty: Optional[float] = None,
    format_penalty: Optional[float] = None,
    tokenizer = None,
    use_warmup: bool = False
) -> dict:
    """Compute detailed breakdown of correctness and length rewards.
    
    Args:
        solution_str: The generated solution text
        ground_truth: The correct answer
        budget: Token budget for the solution, or 'control' for no budget constraint
        budget_window: Window around budget where no penalty is applied (default 0)
        length_penalty: Penalty per token of distance from budget (loads from env if None)
        format_penalty: Penalty for incorrect format (loads from env if None)
        tokenizer: Tokenizer to use (loads from env if None)
        use_warmup: Whether to apply warmup scaling to penalties (default False)
    
    Returns:
        Dictionary containing:
            - extracted_answer: The extracted answer string (or None)
            - correctness_reward: 1.0 if correct, 0.0 otherwise
            - token_count: Number of tokens in solution
            - length_diff: Absolute difference from budget (None for control)
            - length_penalty: Reward/penalty for length (0.0 for control)
            - format_penalty: Reward/penalty for format (0.0 if correct or not used)
            - total_reward: Sum of correctness, length, and format rewards
    """
    # Load from environment if not provided
    if length_penalty is None or tokenizer is None or format_penalty is None:
        _load_from_env()
        if length_penalty is None:
            length_penalty = _LENGTH_PENALTY
        if tokenizer is None:
            tokenizer = _TOKENIZER
        if format_penalty is None:
            format_penalty = _FORMAT_PENALTY
    
    # Apply warmup factor if enabled
    if use_warmup:
        warmup_factor = _get_warmup_factor()
        length_penalty = length_penalty * warmup_factor
        format_penalty = format_penalty * warmup_factor
    
    # Compute correctness reward on the text after the end of the CoT
    if "</think>" not in solution_str:
        correctness_reward = 0.0
        predicted_answer = None
    else:
        output = solution_str.split("</think>")[-1]
        predicted_answer = extract_answer(output, method="flexible")
        
        if predicted_answer is None:
            correctness_reward = 0.0
        else:
            gt_clean = str(ground_truth).replace(',', '').strip()
            
            try:
                # Try numeric comparison
                pred_num = float(predicted_answer)
                gt_num = float(gt_clean)
                correctness_reward = 1.0 if abs(pred_num - gt_num) < 1e-6 else 0.0
            except ValueError:
                # Fallback to string comparison
                correctness_reward = 1.0 if predicted_answer == gt_clean else 0.0
    
    # Compute length metrics
    token_count = len(tokenizer.encode(solution_str))
    
    # Handle control condition (no budget constraint)
    if budget == 'control':
        length_diff = None
        reward_length_penalty = 0.0
    else:
        length_diff = abs(token_count - budget)
        reward_length_penalty = compute_length_penalty(token_count, budget, budget_window, length_penalty)

    # Compute format penalty
    if format_penalty > 0.0:
        if "</think>" in solution_str:
            response = solution_str.split("</think>")[-1]
        else:
            # If no thinking block, the whole string is the response
            response = solution_str
            
        # Check if response matches \boxed{...} with only whitespace around
        if re.fullmatch(r'\s*\\boxed\{[^}]+\}\s*', response, flags=re.DOTALL):
            reward_format_penalty = 0.0
        else:
            reward_format_penalty = -format_penalty
    else:
        reward_format_penalty = 0.0
    
    total_reward = correctness_reward + reward_length_penalty + reward_format_penalty
    
    return {
        'extracted_answer': predicted_answer,
        'correctness_reward': correctness_reward,
        'token_count': token_count,
        'length_diff': length_diff,
        'length_penalty': reward_length_penalty,
        'format_penalty': reward_format_penalty,
        'total_reward': total_reward,
    }


def _log_accumulated_metrics(split: str) -> None:
    """Log accumulated metrics to WandB and reset accumulators."""
    global _ACCUMULATED_METRICS
    
    if not _WANDB_AVAILABLE or wandb.run is None:
        return
    
    if not _ACCUMULATED_METRICS:
        return
    
    # Compute means and log
    prefix = "reward_train" if split == "train" else "reward_test"
    log_dict = {}
    for key, values in _ACCUMULATED_METRICS.items():
        if values:
            log_dict[f"{prefix}/{key}"] = sum(values) / len(values)
    
    if log_dict:
        log_dict[f"{prefix}/n_samples"] = len(_ACCUMULATED_METRICS.get("total_reward", []))
        wandb.log(log_dict, commit=False)
    
    # Reset accumulators
    _ACCUMULATED_METRICS = {}


def _accumulate_metrics(result: dict) -> None:
    """Accumulate metrics from a reward breakdown result."""
    global _ACCUMULATED_METRICS
    
    for key in ["correctness_reward", "token_count", "length_penalty", "format_penalty", "total_reward"]:
        if key not in _ACCUMULATED_METRICS:
            _ACCUMULATED_METRICS[key] = []
        _ACCUMULATED_METRICS[key].append(result[key])
    
    # length_diff can be None for control condition
    if result["length_diff"] is not None:
        if "length_diff" not in _ACCUMULATED_METRICS:
            _ACCUMULATED_METRICS["length_diff"] = []
        _ACCUMULATED_METRICS["length_diff"].append(result["length_diff"])


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict) -> float:
    """Compute total reward score (for RL training).
    
    This is the main interface used during training. It loads configuration from
    environment variables and returns just the total reward.
    """
    global _CALL_COUNT, _CURRENT_SPLIT
    
    if 'budget' not in extra_info:
        raise RuntimeError("Thinking budget not found in extra_info")
    
    current_split = extra_info.get('split')
    
    # Check if split changed - if so, log accumulated metrics from previous split
    if _CURRENT_SPLIT is not None and current_split != _CURRENT_SPLIT:
        _log_accumulated_metrics(_CURRENT_SPLIT)
    _CURRENT_SPLIT = current_split
    
    # Only use warmup during training, not validation
    is_training = current_split == 'train'
    
    if is_training:
        # Increment call count for warmup tracking
        _CALL_COUNT += 1
        
        # Log warmup progress to WandB periodically
        if _WANDB_AVAILABLE and wandb.run is not None and _CALL_COUNT % 1000 == 0:
            warmup_factor = _get_warmup_factor()
            wandb.log({
                "reward/call_count": _CALL_COUNT,
                "reward/warmup_factor": warmup_factor,
            }, commit=False)
    
    budget = extra_info['budget']
    budget_window = extra_info.get('budget_window', 0)
    result = compute_reward_breakdown(solution_str, ground_truth, budget, budget_window=budget_window, use_warmup=is_training)
    
    # Accumulate metrics for WandB logging
    _accumulate_metrics(result)
    
    return result['total_reward']