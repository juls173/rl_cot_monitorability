import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import pandas as pd

def generate_samples(model_path, data_path, output_path, temperature, num_samples=500):
    """Generate reasoning traces at a specific temperature."""
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load data
    df = pd.read_parquet(data_path)
    df = df.head(num_samples)
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['prompt']  # Adjust column name as needed
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate with specific temperature
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        results.append({
            'prompt': prompt,
            'response': response,
            'temperature': temperature,
            'problem_id': idx
        })
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} samples to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--num_samples', type=int, default=500)
    
    args = parser.parse_args()
    generate_samples(args.model_path, args.data_path, args.output_path, 
                     args.temperature, args.num_samples)