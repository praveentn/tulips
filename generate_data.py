"""
Data generation script for creating synthetic training/validation/test datasets.

Generates data for all 4 base tasks based on semantic rules.
"""

import random
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from data_utils import Example, Dataset, split_dataset
from vocab import get_default_vocab
from tasks import get_default_task_registry


def generate_intent_classification_examples(vocab_words, num_examples=100, seed=None):
    """Generate examples for intent classification task."""
    if seed:
        random.seed(seed)
    
    examples = []
    
    # Define rules for each intent
    rules = {
        "COMMAND": lambda: random.choice([
            "YOU GO LEFT",
            "YOU GO RIGHT",
            "YOU GO HERE",
            "YOU GIVE",
            "YOU TAKE HERE",
            "YOU TAKE LEFT",
            "YOU TAKE RIGHT"
        ]),
        "STATEMENT": lambda: random.choice([
            "I GO LEFT",
            "I GO RIGHT",
            "I GO HERE",
            "I GIVE",
            "I TAKE HERE",
            "I TAKE LEFT",
            "I TAKE RIGHT"
        ]),
        "CONFIRMATION": lambda: "YES",
        "NEGATION": lambda: "NO"
    }
    
    # Generate balanced examples
    per_class = num_examples // len(rules)
    
    for label, rule_func in rules.items():
        for _ in range(per_class):
            sentence = rule_func()
            examples.append(Example(sentence, "intent_classification", label))
    
    # Shuffle
    random.shuffle(examples)
    
    return examples


def generate_action_mapping_examples(vocab_words, num_examples=100, seed=None):
    """Generate examples for action mapping task."""
    if seed:
        random.seed(seed)
    
    examples = []
    
    rules = {
        "MOVE_LEFT": lambda: random.choice([
            "GO LEFT",
            "YOU GO LEFT",
            "I GO LEFT"
        ]),
        "MOVE_RIGHT": lambda: random.choice([
            "GO RIGHT",
            "YOU GO RIGHT",
            "I GO RIGHT"
        ]),
        "MOVE_HERE": lambda: random.choice([
            "GO HERE",
            "YOU GO HERE",
            "I GO HERE"
        ]),
        "GIVE_ITEM": lambda: random.choice([
            "GIVE",
            "YOU GIVE",
            "I GIVE",
            "GIVE HERE"
        ]),
        "TAKE_ITEM": lambda: random.choice([
            "TAKE",
            "YOU TAKE",
            "I TAKE",
            "TAKE HERE"
        ]),
        "STAY": lambda: random.choice([
            "HERE",
            "I HERE",
            "YOU HERE"
        ])
    }
    
    per_class = num_examples // len(rules)
    
    for label, rule_func in rules.items():
        for _ in range(per_class):
            sentence = rule_func()
            examples.append(Example(sentence, "action_mapping", label))
    
    random.shuffle(examples)
    
    return examples


def generate_response_generation_examples(vocab_words, num_examples=100, seed=None):
    """Generate examples for response generation task."""
    if seed:
        random.seed(seed)
    
    examples = []
    
    # Input -> Expected Response mappings
    patterns = [
        ("YOU GO LEFT", "I_GO_LEFT"),
        ("YOU GO RIGHT", "I_GO_RIGHT"),
        ("YOU GIVE", "I_TAKE"),
        ("I GIVE", "YOU_TAKE"),
        ("I TAKE", "YOU_GIVE"),
        ("YOU TAKE", "I_GIVE"),
        ("GO LEFT", "YES"),
        ("GO RIGHT", "YES"),
        ("TAKE HERE", "YES"),
        ("GIVE HERE", "NO"),
        ("HERE", "SILENT"),
        ("I HERE", "SILENT"),
    ]
    
    per_pattern = num_examples // len(patterns)
    
    for input_text, response in patterns:
        for _ in range(per_pattern):
            examples.append(Example(input_text, "response_generation", response))
    
    random.shuffle(examples)
    
    return examples


def generate_direction_detection_examples(vocab_words, num_examples=100, seed=None):
    """Generate examples for direction detection task."""
    if seed:
        random.seed(seed)
    
    examples = []
    
    rules = {
        "LEFT": lambda: random.choice([
            "GO LEFT",
            "YOU GO LEFT",
            "I GO LEFT",
            "TAKE LEFT",
            "LEFT"
        ]),
        "RIGHT": lambda: random.choice([
            "GO RIGHT",
            "YOU GO RIGHT",
            "I GO RIGHT",
            "TAKE RIGHT",
            "RIGHT"
        ]),
        "HERE": lambda: random.choice([
            "GO HERE",
            "YOU GO HERE",
            "I GO HERE",
            "TAKE HERE",
            "GIVE HERE",
            "HERE"
        ]),
        "NONE": lambda: random.choice([
            "YES",
            "NO",
            "I GIVE",
            "YOU TAKE",
            "GIVE",
            "TAKE"
        ])
    }
    
    per_class = num_examples // len(rules)
    
    for label, rule_func in rules.items():
        for _ in range(per_class):
            sentence = rule_func()
            examples.append(Example(sentence, "direction_detection", label))
    
    random.shuffle(examples)
    
    return examples


def generate_all_datasets(output_dir="data/tasks", seed=42):
    """
    Generate all datasets for all tasks.
    
    Args:
        output_dir: Directory to save datasets
        seed: Random seed for reproducibility
    """
    print("Generating synthetic datasets for all tasks...")
    print("="*70)
    
    # Get vocabulary
    vocab = get_default_vocab()
    vocab_words = vocab.words
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define tasks and their generators
    task_generators = {
        "intent_classification": generate_intent_classification_examples,
        "action_mapping": generate_action_mapping_examples,
        "response_generation": generate_response_generation_examples,
        "direction_detection": generate_direction_detection_examples
    }
    
    # Generate data for each task
    for task_name, generator in task_generators.items():
        print(f"\n{task_name}:")
        print("-" * 50)
        
        # Generate examples
        examples = generator(vocab_words, num_examples=200, seed=seed)
        print(f"  Generated {len(examples)} examples")
        
        # Split into train/val/test
        train_dataset, val_dataset, test_dataset = split_dataset(
            examples,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=seed
        )
        
        # Create task directory
        task_dir = output_path / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        train_dataset.save_jsonl(str(task_dir / "train.jsonl"))
        val_dataset.save_jsonl(str(task_dir / "val.jsonl"))
        test_dataset.save_jsonl(str(task_dir / "test.jsonl"))
        
        print(f"  ✓ Saved train/val/test splits to {task_dir}")
    
    print("\n" + "="*70)
    print("Dataset generation complete!")
    print(f"All datasets saved to: {output_path}")
    
    # Print summary
    print("\nSummary:")
    for task_name in task_generators.keys():
        task_dir = output_path / task_name
        train_size = len(Dataset.load_jsonl(str(task_dir / "train.jsonl")))
        val_size = len(Dataset.load_jsonl(str(task_dir / "val.jsonl")))
        test_size = len(Dataset.load_jsonl(str(task_dir / "test.jsonl")))
        print(f"  {task_name}: train={train_size}, val={val_size}, test={test_size}")


if __name__ == "__main__":
    # Generate datasets relative to the script location
    script_dir = Path(__file__).parent.parent
    output_dir = script_dir / "data" / "tasks"
    
    generate_all_datasets(output_dir=str(output_dir), seed=42)
