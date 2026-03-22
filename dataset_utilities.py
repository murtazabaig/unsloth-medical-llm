"""
Medical Dataset Utilities
==========================

Utilities for preparing and managing medical datasets for fine-tuning.
Includes data loading, validation, and conversion functions.
"""

import pandas as pd
import json
import os
from datasets import Dataset, load_dataset
from typing import List, Dict, Optional, Tuple
import argparse


class MedicalDatasetManager:
    """Manager for medical dataset operations"""
    
    @staticmethod
    def create_from_csv(csv_path: str) -> Dataset:
        """
        Create Dataset from CSV file
        
        Args:
            csv_path: Path to CSV file with 'question' and 'answer' columns
        
        Returns:
            HuggingFace Dataset object
        """
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Validate columns
        required_cols = ['question', 'answer']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        dataset = Dataset.from_pandas(df[required_cols])
        print(f"✓ Loaded {len(dataset)} examples")
        
        return dataset
    
    @staticmethod
    def create_from_json(json_path: str) -> Dataset:
        """
        Create Dataset from JSONL file
        
        Args:
            json_path: Path to JSONL file (one JSON object per line)
        
        Returns:
            HuggingFace Dataset object
        """
        print(f"Loading dataset from {json_path}...")
        
        data = []
        with open(json_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        print(f"✓ Loaded {len(dataset)} examples")
        
        return dataset
    
    @staticmethod
    def create_from_dict_list(data_list: List[Dict]) -> Dataset:
        """
        Create Dataset from list of dictionaries
        
        Args:
            data_list: List of dictionaries with 'question' and 'answer' keys
        
        Returns:
            HuggingFace Dataset object
        """
        df = pd.DataFrame(data_list)
        dataset = Dataset.from_pandas(df)
        print(f"✓ Created dataset with {len(dataset)} examples")
        
        return dataset
    
    @staticmethod
    def load_from_huggingface(dataset_name: str, split: str = "train",
                              sample_size: Optional[int] = None) -> Dataset:
        """
        Load dataset from HuggingFace Hub
        
        Args:
            dataset_name: Name of dataset on HuggingFace Hub
            split: Dataset split to load
            sample_size: Optional number of samples to take
        
        Returns:
            HuggingFace Dataset object
        """
        print(f"Loading {dataset_name} from HuggingFace Hub...")
        
        dataset = load_dataset(dataset_name, split=split)
        
        if sample_size:
            dataset = dataset.select(range(min(sample_size, len(dataset))))
        
        print(f"✓ Loaded {len(dataset)} examples")
        
        return dataset
    
    @staticmethod
    def save_to_csv(dataset: Dataset, output_path: str):
        """
        Save Dataset to CSV file
        
        Args:
            dataset: HuggingFace Dataset object
            output_path: Path to save CSV
        """
        df = dataset.to_pandas()
        df.to_csv(output_path, index=False)
        print(f"✓ Dataset saved to {output_path}")
    
    @staticmethod
    def save_to_json(dataset: Dataset, output_path: str):
        """
        Save Dataset to JSONL file
        
        Args:
            dataset: HuggingFace Dataset object
            output_path: Path to save JSONL
        """
        with open(output_path, 'w') as f:
            for example in dataset:
                f.write(json.dumps(example) + '\n')
        print(f"✓ Dataset saved to {output_path}")
    
    @staticmethod
    def split_dataset(dataset: Dataset, train_ratio: float = 0.8,
                     seed: int = 42) -> Tuple[Dataset, Dataset]:
        """
        Split dataset into train and evaluation sets
        
        Args:
            dataset: Dataset to split
            train_ratio: Ratio for training set
            seed: Random seed
        
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        split = dataset.train_test_split(
            test_size=1 - train_ratio,
            seed=seed
        )
        
        train_dataset = split['train']
        eval_dataset = split['test']
        
        print(f"✓ Split dataset:")
        print(f"  Train: {len(train_dataset)} examples ({train_ratio*100:.0f}%)")
        print(f"  Eval: {len(eval_dataset)} examples ({(1-train_ratio)*100:.0f}%)")
        
        return train_dataset, eval_dataset
    
    @staticmethod
    def validate_dataset(dataset: Dataset) -> Dict[str, any]:
        """
        Validate dataset structure and content
        
        Args:
            dataset: Dataset to validate
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "total_examples": len(dataset),
            "columns": dataset.column_names,
            "has_questions": False,
            "has_answers": False,
            "avg_question_length": 0,
            "avg_answer_length": 0,
            "min_question_length": float('inf'),
            "max_question_length": 0,
            "min_answer_length": float('inf'),
            "max_answer_length": 0,
        }
        
        question_lengths = []
        answer_lengths = []
        
        for example in dataset:
            if 'question' in example and example['question']:
                results["has_questions"] = True
                q_len = len(example['question'].split())
                question_lengths.append(q_len)
                results["min_question_length"] = min(
                    results["min_question_length"], q_len
                )
                results["max_question_length"] = max(
                    results["max_question_length"], q_len
                )
            
            if 'answer' in example and example['answer']:
                results["has_answers"] = True
                a_len = len(example['answer'].split())
                answer_lengths.append(a_len)
                results["min_answer_length"] = min(
                    results["min_answer_length"], a_len
                )
                results["max_answer_length"] = max(
                    results["max_answer_length"], a_len
                )
        
        if question_lengths:
            results["avg_question_length"] = sum(question_lengths) / len(question_lengths)
        if answer_lengths:
            results["avg_answer_length"] = sum(answer_lengths) / len(answer_lengths)
        
        return results
    
    @staticmethod
    def print_validation_report(validation_results: Dict):
        """Print validation results"""
        print("\nDataset Validation Report")
        print("=" * 60)
        print(f"Total Examples: {validation_results['total_examples']}")
        print(f"Columns: {', '.join(validation_results['columns'])}")
        print(f"Has Questions: {'✓' if validation_results['has_questions'] else '✗'}")
        print(f"Has Answers: {'✓' if validation_results['has_answers'] else '✗'}")
        
        print("\nQuestion Statistics:")
        print(f"  Average Length: {validation_results['avg_question_length']:.0f} words")
        print(f"  Min Length: {validation_results['min_question_length']} words")
        print(f"  Max Length: {validation_results['max_question_length']} words")
        
        print("\nAnswer Statistics:")
        print(f"  Average Length: {validation_results['avg_answer_length']:.0f} words")
        print(f"  Min Length: {validation_results['min_answer_length']} words")
        print(f"  Max Length: {validation_results['max_answer_length']} words")
        print("=" * 60)
    
    @staticmethod
    def create_sample_medical_dataset() -> Dataset:
        """
        Create a sample medical dataset for testing
        
        Returns:
            Sample Dataset with medical Q&A pairs
        """
        samples = [
            {
                "question": "What are the early warning signs of myocardial infarction?",
                "answer": "Early warning signs of MI include chest pain or pressure, shortness of breath, sweating, nausea, and arm or neck pain. Women may experience atypical symptoms like fatigue or jaw pain."
            },
            {
                "question": "How is acute stroke managed?",
                "answer": "Acute stroke management involves rapid diagnosis via CT/MRI to rule out hemorrhage, thrombolytics (alteplase) within 3-4.5 hours, endovascular thrombectomy for large vessel occlusion, and supportive care."
            },
            {
                "question": "What is the treatment for type 2 diabetes?",
                "answer": "Type 2 diabetes treatment includes lifestyle modifications, metformin as first-line medication, and additional agents like sulfonylureas, DPP-4 inhibitors, SGLT2 inhibitors, or GLP-1 agonists as needed."
            },
            {
                "question": "Describe the clinical presentation of pneumonia.",
                "answer": "Pneumonia presents with cough (productive or dry), fever, dyspnea, pleuritic chest pain, and fatigue. Findings include crackles on auscultation and infiltrates on chest imaging."
            },
            {
                "question": "What is the management of sepsis?",
                "answer": "Sepsis management includes early recognition, blood cultures before antibiotics, broad-spectrum antibiotics within 1 hour, IV fluid resuscitation, vasopressors for hypotension, and source control."
            },
        ]
        
        return MedicalDatasetManager.create_from_dict_list(samples)


def main():
    """Command-line interface for dataset management"""
    parser = argparse.ArgumentParser(description="Medical Dataset Manager")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert between formats")
    convert_parser.add_argument("--input", required=True, help="Input file path")
    convert_parser.add_argument("--output", required=True, help="Output file path")
    convert_parser.add_argument("--from", dest="format_from", 
                               choices=["csv", "json"], default="csv")
    convert_parser.add_argument("--to", dest="format_to",
                               choices=["csv", "json"], default="json")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("--input", required=True, help="Input file path")
    validate_parser.add_argument("--format", choices=["csv", "json"], default="csv")
    
    # Split command
    split_parser = subparsers.add_parser("split", help="Split dataset")
    split_parser.add_argument("--input", required=True, help="Input file path")
    split_parser.add_argument("--output_dir", required=True, help="Output directory")
    split_parser.add_argument("--train_ratio", type=float, default=0.8)
    split_parser.add_argument("--format", choices=["csv", "json"], default="csv")
    
    # Create sample command
    sample_parser = subparsers.add_parser("sample", help="Create sample dataset")
    sample_parser.add_argument("--output", required=True, help="Output file path")
    sample_parser.add_argument("--format", choices=["csv", "json"], default="csv")
    
    args = parser.parse_args()
    
    if args.command == "convert":
        # Load
        if args.format_from == "csv":
            dataset = MedicalDatasetManager.create_from_csv(args.input)
        else:
            dataset = MedicalDatasetManager.create_from_json(args.input)
        
        # Save
        if args.format_to == "csv":
            MedicalDatasetManager.save_to_csv(dataset, args.output)
        else:
            MedicalDatasetManager.save_to_json(dataset, args.output)
    
    elif args.command == "validate":
        # Load
        if args.format == "csv":
            dataset = MedicalDatasetManager.create_from_csv(args.input)
        else:
            dataset = MedicalDatasetManager.create_from_json(args.input)
        
        # Validate
        results = MedicalDatasetManager.validate_dataset(dataset)
        MedicalDatasetManager.print_validation_report(results)
    
    elif args.command == "split":
        # Load
        if args.format == "csv":
            dataset = MedicalDatasetManager.create_from_csv(args.input)
        else:
            dataset = MedicalDatasetManager.create_from_json(args.input)
        
        # Split
        os.makedirs(args.output_dir, exist_ok=True)
        train, eval = MedicalDatasetManager.split_dataset(
            dataset, args.train_ratio
        )
        
        # Save
        if args.format == "csv":
            MedicalDatasetManager.save_to_csv(
                train, os.path.join(args.output_dir, "train.csv")
            )
            MedicalDatasetManager.save_to_csv(
                eval, os.path.join(args.output_dir, "eval.csv")
            )
        else:
            MedicalDatasetManager.save_to_json(
                train, os.path.join(args.output_dir, "train.jsonl")
            )
            MedicalDatasetManager.save_to_json(
                eval, os.path.join(args.output_dir, "eval.jsonl")
            )
    
    elif args.command == "sample":
        # Create sample
        dataset = MedicalDatasetManager.create_sample_medical_dataset()
        
        # Save
        if args.format == "csv":
            MedicalDatasetManager.save_to_csv(dataset, args.output)
        else:
            MedicalDatasetManager.save_to_json(dataset, args.output)


if __name__ == "__main__":
    main()
