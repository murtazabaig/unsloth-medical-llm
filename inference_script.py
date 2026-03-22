"""
Medical QLoRA Model Inference Script
=====================================

This script demonstrates how to use the fine-tuned medical LoRA adapter
for inference outside of the Jupyter notebook.

Usage:
    python inference_script.py --adapter_path <path_to_adapter> --query "medical question"
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModel
import sys


class MedicalModelInference:
    """Wrapper for medical model inference with LoRA adapter"""
    
    def __init__(self, adapter_path, device="auto", dtype="auto"):
        """
        Initialize the medical model with LoRA adapter
        
        Args:
            adapter_path: Path to the saved LoRA adapter
            device: Device to load model on ('cuda', 'cpu', 'auto')
            dtype: Data type ('auto', 'float16', 'float32')
        """
        print(f"Loading adapter from {adapter_path}...")
        
        # Load adapter and model
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_path,
            device_map=device,
            torch_dtype=dtype,
            load_in_4bit=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # Set model to eval mode
        self.model.eval()
        
        print("✓ Model loaded successfully!")
    
    def get_medical_prompt(self, question):
        """Format medical question as per training format"""
        prompt = f"""Below is a medical question and its answer. Provide a clear, accurate response.

### Question:
{question}

### Answer:"""
        return prompt
    
    def inference(
        self,
        question,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        **kwargs
    ):
        """
        Run inference on a medical question
        
        Args:
            question: Medical question string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
        
        Returns:
            Generated answer string
        """
        # Format prompt
        prompt = self.get_medical_prompt(question)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate without gradients
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode output
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract answer (after "### Answer:")
        if "### Answer:" in response:
            answer = response.split("### Answer:")[1].strip()
        else:
            answer = response
        
        return answer
    
    def batch_inference(self, questions, **kwargs):
        """
        Run inference on multiple questions
        
        Args:
            questions: List of medical questions
            **kwargs: Parameters passed to inference()
        
        Returns:
            List of generated answers
        """
        answers = []
        for i, question in enumerate(questions, 1):
            print(f"\nProcessing query {i}/{len(questions)}...")
            answer = self.inference(question, **kwargs)
            answers.append(answer)
        
        return answers


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Medical LoRA Model Inference"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the saved LoRA adapter"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single medical question to ask"
    )
    parser.add_argument(
        "--query_file",
        type=str,
        help="File with medical questions (one per line)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling p (default: 0.95)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Initialize model
    try:
        model = MedicalModelInference(
            args.adapter_path,
            device=args.device,
            dtype="auto"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run inference
    if args.query:
        # Single query
        print(f"\nQuestion: {args.query}")
        print("-" * 60)
        answer = model.inference(
            args.query,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(f"Answer: {answer}")
        
    elif args.query_file:
        # Multiple queries from file
        try:
            with open(args.query_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            print(f"\nLoaded {len(queries)} queries from {args.query_file}")
            print("=" * 60)
            
            answers = model.batch_inference(
                queries,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            # Save results
            results_file = "inference_results.txt"
            with open(results_file, 'w') as f:
                for q, a in zip(queries, answers):
                    f.write(f"Question: {q}\n")
                    f.write(f"Answer: {a}\n")
                    f.write("-" * 60 + "\n\n")
            
            print(f"\n✓ Results saved to {results_file}")
            
        except FileNotFoundError:
            print(f"Error: Could not find file {args.query_file}")
            sys.exit(1)
    
    else:
        # Interactive mode
        print("\nMedical Model Inference - Interactive Mode")
        print("Type 'quit' or 'exit' to stop\n")
        print("=" * 60)
        
        while True:
            try:
                query = input("\nEnter medical question: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    print("Exiting...")
                    break
                
                if not query:
                    continue
                
                print("-" * 60)
                answer = model.inference(
                    query,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                print(f"Answer: {answer}")
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue


if __name__ == "__main__":
    main()
