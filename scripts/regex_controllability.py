import os
import json
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from gsm8k_reward import extract_answer

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 7)
plt.rcParams['font.size'] = 11


class ReadabilityDriftEvaluator:
    """Evaluate linguistic drift through comma density calculation"""
    
    def __init__(self, base_model_path: str, model_configs: List[Dict],
                 batch_size: int = 8):
        self.base_model_path = base_model_path
        self.model_configs = model_configs
        self.batch_size = batch_size
        self.tokenizer = None
        
    def load_tokenizer(self):
        """Load tokenizer once"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_model(self, checkpoint_path: str = None):
        """Load a single model (base or checkpoint)"""
        if checkpoint_path is None:
            print(f"Loading base model: {self.base_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            print(f"Loading checkpoint: {checkpoint_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        model.eval()
        return model
    
    def unload_model(self, model):
        """Unload model from memory"""
        del model
        torch.cuda.empty_cache()
    
    def extract_cot(self, text: str) -> str:
        """Extract chain of thought (everything before </think>)"""
        if '</think>' in text:
            cot = text.split('</think>')[0]
            if '<think>' in cot:
                cot = cot.split('<think>')[-1]
            return cot.strip()
        return text.strip()
    
    def calculate_comma_density(self, cot: str) -> float:
        """Calculate comma density as (number of commas / CoT length) * 100"""
        if len(cot) == 0:
            return 0.0
        
        comma_count = cot.count(',')
        # Return as percentage (commas per 100 characters)
        comma_density = (comma_count / len(cot)) * 100
        
        return comma_density
    
    def generate_cots_batched(self, questions: List[str], model, model_name: str,
                              max_new_tokens: int = 512, temperature: float = 1.0,
                              use_no_comma_prompt: bool = False) -> tuple[List[str], List[str]]:
        """Generate chain of thoughts from a model using batched inference"""
        all_cots = []
        all_full_outputs = []
        
        for i in tqdm(range(0, len(questions), self.batch_size), 
                     desc=f"Generating CoTs from {model_name}"):
            batch_questions = questions[i:i + self.batch_size]
            
            if use_no_comma_prompt:
                prompts = [f'<｜User｜>{q}<｜Assistant｜>Let\'s think step by step and output the final answer after "####". Do not use any commas in your private reasoning chain. <think>' 
                          for q in batch_questions]
            else:
                prompts = [f'<｜User｜>{q}<｜Assistant｜>Let\'s think step by step and output the final answer after "####". <think>' 
                          for q in batch_questions]
            
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            outputs = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            
            for prompt, output in zip(prompts, outputs):
                output = output[len(prompt):].strip()
                all_full_outputs.append(output)
                cot = self.extract_cot(output)
                all_cots.append(cot)
        
        return all_cots, all_full_outputs
    
    def evaluate_correctness(self, full_outputs: List[str], 
                           ground_truths: List[str]) -> Dict:
        """Evaluate answer correctness"""
        correct = 0
        total = 0
        
        for output, gt in zip(full_outputs, ground_truths):
            predicted = extract_answer(output, method="flexible")
            
            if predicted is not None and gt is not None:
                total += 1
                try:
                    pred_num = float(predicted.replace(',', ''))
                    gt_num = float(str(gt).replace(',', ''))
                    if abs(pred_num - gt_num) < 1e-5:
                        correct += 1
                except:
                    if predicted.strip() == str(gt).strip():
                        correct += 1
        
        accuracy = float((correct / total * 100) if total > 0 else 0)
        return {
            'accuracy': accuracy,
            'correct': int(correct),
            'total': int(total)
        }
    
    def evaluate_model_comma_density(self, cots: List[str], full_outputs: List[str], 
                                     ground_truths: List[str], model_name: str) -> Dict:
        """Evaluate comma density for all CoTs from a model"""
        
        print(f"\nCalculating comma density for {model_name}...")
        
        densities = []
        for cot, full_output, gt in tqdm(zip(cots, full_outputs, ground_truths), 
                                         total=len(cots), desc="Calculating comma density"):
            if len(cot.strip()) < 10:
                continue
            
            # Calculate comma density
            comma_density = self.calculate_comma_density(cot)
            densities.append(comma_density)
        
        return {
            'mean_comma_density': float(np.mean(densities)) if densities else 0.0,
            'std_comma_density': float(np.std(densities)) if densities else 0.0,
            'median_comma_density': float(np.median(densities)) if densities else 0.0,
            'min_comma_density': float(np.min(densities)) if densities else 0.0,
            'max_comma_density': float(np.max(densities)) if densities else 0.0,
            'num_samples': int(len(densities)),
            'all_densities': [float(d) for d in densities],
            'generated_samples': full_outputs
        }
    
    def plot_comma_prevalence_comparison(self, all_model_results: Dict[str, Dict], 
                                         output_path: str = 'comma_prevalence_comparison.png'):
        """Plot comma density for all models with and without no-comma prompt"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Extract temperature from model name for legend
        def get_temp_label(model_name):
            if 'temp09' in model_name:
                return 'Temp 0.9'
            elif 'temp11' in model_name:
                return 'Temp 1.1'
            elif 'temp13' in model_name:
                return 'Temp 1.3'
            elif 'temp15' in model_name:
                return 'Temp 1.5'
            return model_name
        
        # Prepare data for plotting
        model_names = []
        normal_means = []
        normal_stds = []
        no_comma_means = []
        no_comma_stds = []
        accuracies_normal = []
        accuracies_no_comma = []
        
        for idx, (model_name, results) in enumerate(sorted(all_model_results.items())):
            model_names.append(get_temp_label(model_name))
            
            normal_means.append(results['normal']['mean_comma_density'])
            normal_stds.append(results['normal']['std_comma_density'])
            no_comma_means.append(results['no_comma']['mean_comma_density'])
            no_comma_stds.append(results['no_comma']['std_comma_density'])
            
            accuracies_normal.append(results['normal'].get('accuracy', 0))
            accuracies_no_comma.append(results['no_comma'].get('accuracy', 0))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        # Plot 1: Comma Density
        bars1 = ax1.bar(x - width/2, normal_means, width, yerr=normal_stds,
                        label='Normal Prompt', color='#2E86AB', alpha=0.8, capsize=5)
        bars2 = ax1.bar(x + width/2, no_comma_means, width, yerr=no_comma_stds,
                        label='No-Comma Prompt', color='#F18F01', alpha=0.8, capsize=5)
        
        ax1.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Comma Density (commas per 100 chars)', fontsize=13, fontweight='bold')
        ax1.set_title('Comma Density: Normal vs No-Comma Prompt',
                     fontsize=15, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=0)
        ax1.legend(fontsize=11, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Accuracy
        bars3 = ax2.bar(x - width/2, accuracies_normal, width,
                        label='Normal Prompt', color='#2E86AB', alpha=0.8)
        bars4 = ax2.bar(x + width/2, accuracies_no_comma, width,
                        label='No-Comma Prompt', color='#F18F01', alpha=0.8)
        
        ax2.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Answer Correctness: Normal vs No-Comma Prompt',
                     fontsize=15, fontweight='bold', pad=15)
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=0)
        ax2.legend(fontsize=11, loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {output_path}")
        plt.close()


def main():
    """Main evaluation pipeline for comma density across multiple models"""
    
    BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    BASE_CHECKPOINT_DIR = "/workspace/checkpoints/verl_gsm8k_v2"
    
    # Define all 4 models with their final checkpoints only
    MODEL_CONFIGS = [
        {
            'name': 'r1qwen15b_grpo_temp09_topp95_entropy0005_kl0001',
            'final_checkpoint': 116
        },
        {
            'name': 'r1qwen15b_grpo_temp11_topp95_entropy0005_kl0001',
            'final_checkpoint': 116
        },
        {
            'name': 'r1qwen15b_grpo_temp13_topp95_entropy0005_kl0001',
            'final_checkpoint': 116
        },
        {
            'name': 'r1qwen15b_grpo_temp15_topp95_entropy0005_kl0001',
            'final_checkpoint': 116
        }
    ]
    
    GSM8K_PATH = "/workspace/data/gsm8k/test.parquet"
    BATCH_SIZE = 32
    
    # Load data
    print("Loading GSM8K test data...")
    df = pd.read_parquet(GSM8K_PATH)
    questions = [row['prompt'][0]['content'] for _, row in df.iterrows()]
    ground_truths = [row['reward_model']['ground_truth'] for _, row in df.iterrows()]
    
    num_samples = min(100, len(questions))
    sampled_questions = questions[:num_samples]
    sampled_ground_truths = ground_truths[:num_samples]
    print(f"Evaluating on {num_samples} questions")
    
    # Initialize evaluator
    evaluator = ReadabilityDriftEvaluator(
        BASE_MODEL,
        MODEL_CONFIGS,
        batch_size=BATCH_SIZE
    )
    evaluator.load_tokenizer()
    
    # Store results for all models
    all_model_results = {}
    
    # Helper function to get temperature from model name
    def get_temp_value(model_name):
        if 'temp09' in model_name:
            return 0.9
        elif 'temp11' in model_name:
            return 1.1
        elif 'temp13' in model_name:
            return 1.3
        elif 'temp15' in model_name:
            return 1.5
        return 1.0
    
    # Evaluate each model
    for model_config in MODEL_CONFIGS:
        model_name = model_config['name']
        final_checkpoint = model_config['final_checkpoint']
        
        print(f"\n{'='*70}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"{'='*70}")
        
        checkpoint_path = f"{BASE_CHECKPOINT_DIR}/{model_name}/global_step_{final_checkpoint}/actor/lora_adapter"
        
        # Load model once
        model = evaluator.load_model(checkpoint_path)
        temp = get_temp_value(model_name)
        
        model_results = {}
        
        # Evaluate with NORMAL prompt
        print(f"\n{'-'*60}")
        print(f"Evaluating with NORMAL prompt")
        print(f"{'-'*60}")
        
        cots_normal, full_outputs_normal = evaluator.generate_cots_batched(
            sampled_questions, model, f"{model_name}_normal", 
            temperature=temp, use_no_comma_prompt=False
        )
        
        correctness_normal = evaluator.evaluate_correctness(
            full_outputs_normal, sampled_ground_truths
        )
        print(f"Accuracy (Normal): {correctness_normal['accuracy']:.2f}% "
              f"({correctness_normal['correct']}/{correctness_normal['total']})")
        
        results_normal = evaluator.evaluate_model_comma_density(
            cots_normal, full_outputs_normal, sampled_ground_truths, f"{model_name}_normal"
        )
        results_normal.update(correctness_normal)
        model_results['normal'] = results_normal
        
        print(f"\nResults (Normal):")
        print(f"  Mean Comma Density: {results_normal['mean_comma_density']:.4f} commas/100 chars")
        print(f"  Std Dev: {results_normal['std_comma_density']:.4f}")
        
        # Evaluate with NO-COMMA prompt
        print(f"\n{'-'*60}")
        print(f"Evaluating with NO-COMMA prompt")
        print(f"{'-'*60}")
        
        cots_no_comma, full_outputs_no_comma = evaluator.generate_cots_batched(
            sampled_questions, model, f"{model_name}_no_comma", 
            temperature=temp, use_no_comma_prompt=True
        )
        
        correctness_no_comma = evaluator.evaluate_correctness(
            full_outputs_no_comma, sampled_ground_truths
        )
        print(f"Accuracy (No-Comma): {correctness_no_comma['accuracy']:.2f}% "
              f"({correctness_no_comma['correct']}/{correctness_no_comma['total']})")
        
        results_no_comma = evaluator.evaluate_model_comma_density(
            cots_no_comma, full_outputs_no_comma, sampled_ground_truths, f"{model_name}_no_comma"
        )
        results_no_comma.update(correctness_no_comma)
        model_results['no_comma'] = results_no_comma
        
        print(f"\nResults (No-Comma):")
        print(f"  Mean Comma Density: {results_no_comma['mean_comma_density']:.4f} commas/100 chars")
        print(f"  Std Dev: {results_no_comma['std_comma_density']:.4f}")
        
        # Unload model to free memory
        evaluator.unload_model(model)
        
        all_model_results[model_name] = model_results
    
    # Save all results
    output_data = {
        'evaluation_config': {
            'base_model': BASE_MODEL,
            'num_samples': int(num_samples),
            'batch_size': BATCH_SIZE,
            'models': MODEL_CONFIGS
        },
        'results': all_model_results
    }
    
    with open('comma_density_results_final_models.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print("\n" + "="*70)
    print("Results saved to: comma_density_results_final_models.json")
    
    # Plot comparison
    evaluator.plot_comma_prevalence_comparison(all_model_results, 'comma_density_comparison.png')
    
    print("="*70)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()