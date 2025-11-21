import os
import re
from typing import Literal, Optional
from transformers import AutoTokenizer

# Global variables (lazy-loaded from environment if not passed as arguments)
_LENGTH_PENALTY = None
_MODEL_NAME = None
_TOKENIZER = None
_FORMAT_PENALTY = None

def _load_from_env():
    """Lazy-load configuration from environment variables."""
    global _LENGTH_PENALTY, _MODEL_NAME, _TOKENIZER, _FORMAT_PENALTY
    if _LENGTH_PENALTY is None:
        _LENGTH_PENALTY = float(os.environ["LENGTH_PENALTY"])
    if _MODEL_NAME is None:
        _MODEL_NAME = os.environ["TOKENIZER_MODEL_NAME"]
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)
    if _FORMAT_PENALTY is None:
        _FORMAT_PENALTY = float(os.environ.get("FORMAT_ONLY_ANSWER_PENALTY", 0.0))

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
    length_penalty: Optional[float] = None,
    format_penalty: Optional[float] = None,
    tokenizer = None
) -> dict:
    """Compute detailed breakdown of correctness and length rewards.
    
    Args:
        solution_str: The generated solution text
        ground_truth: The correct answer
        budget: Token budget for the solution, or 'control' for no budget constraint
        length_penalty: Penalty per token of distance from budget (loads from env if None)
        format_penalty: Penalty for incorrect format (loads from env if None)
        tokenizer: Tokenizer to use (loads from env if None)
    
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
        reward_length_penalty = -length_penalty * length_diff

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


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict) -> float:
    """Compute total reward score (for RL training).
    
    This is the main interface used during training. It loads configuration from
    environment variables and returns just the total reward.
    """
    if 'budget' not in extra_info:
        raise RuntimeError("Thinking budget not found in extra_info")
    
    budget = extra_info['budget']
    result = compute_reward_breakdown(solution_str, ground_truth, budget)
    return result['total_reward']