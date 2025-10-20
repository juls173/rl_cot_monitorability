import os
import re
from typing import Literal, Optional
from transformers import AutoTokenizer

# Load configuration from environment variables (once at module load)
LENGTH_REWARD = float(os.environ["LENGTH_REWARD"])
LENGTH_EXPONENT = float(os.environ["LENGTH_EXPONENT"])
MODEL_NAME = os.environ["TOKENIZER_MODEL_NAME"]
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

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

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict) -> float:
    # Compute correctness reward
    predicted_answer = extract_answer(solution_str, method="flexible")
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
    
    # Compute length reward
    # Reward is LENGTH_REWARD * min(1, (100 / |token_count - budget|)^LENGTH_EXPONENT)
    length_bonus = 0.0
    if 'budget' in extra_info:
        token_count = len(TOKENIZER.encode(solution_str))
        budget = extra_info['budget']
        if LENGTH_EXPONENT == float('inf'):
            if abs(token_count - budget) <= 100:
                length_bonus = LENGTH_REWARD
        else:
            difference = abs(token_count - budget)
            if difference <= 100:
                length_bonus = LENGTH_REWARD
            else:
                length_bonus = LENGTH_REWARD * (100 / difference) ** LENGTH_EXPONENT
    else:
        raise RuntimeError("Thinking budget not found in extra_info")
    
    return correctness_reward + length_bonus