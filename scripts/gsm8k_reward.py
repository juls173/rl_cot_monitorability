import re
from typing import Literal, Optional

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

    predicted_answer = extract_answer(solution_str, method="flexible")
    if predicted_answer is None:
        return 0.0
    
    gt_clean = str(ground_truth).replace(',', '').strip()
    
    try:
        # Try numeric comparison
        pred_num = float(predicted_answer)
        gt_num = float(gt_clean)
        return 1.0 if abs(pred_num - gt_num) < 1e-6 else 0.0
    except ValueError:
        # Fallback to string comparison
        return 1.0 if predicted_answer == gt_clean else 0.0