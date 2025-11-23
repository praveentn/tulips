"""
Data utilities for loading, managing, and splitting datasets.

Handles JSONL files for training, validation, and test sets.
"""

import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import random


@dataclass
class Example:
    """A single training/validation/test example."""
    input_text: str
    task: str
    label: str
    input_indices: Optional[List[int]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "input": self.input_text,
            "task": self.task,
            "label": self.label
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Example':
        """Create from dictionary."""
        return cls(
            input_text=d["input"],
            task=d["task"],
            label=d["label"]
        )


class Dataset:
    """Container for a dataset (train/val/test)."""
    
    def __init__(self, examples: List[Example], name: str = "dataset"):
        """
        Initialize dataset.
        
        Args:
            examples: List of Example objects
            name: Name of the dataset (e.g., "train", "val", "test")
        """
        self.examples = examples
        self.name = name
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Example:
        """Get example by index."""
        return self.examples[idx]
    
    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle examples in place."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.examples)
    
    def get_task_counts(self) -> Dict[str, int]:
        """Get count of examples per task."""
        counts = {}
        for ex in self.examples:
            counts[ex.task] = counts.get(ex.task, 0) + 1
        return counts
    
    def get_label_counts(self, task: Optional[str] = None) -> Dict[str, int]:
        """
        Get count of examples per label.
        
        Args:
            task: If specified, only count labels for this task
        """
        counts = {}
        for ex in self.examples:
            if task is None or ex.task == task:
                counts[ex.label] = counts.get(ex.label, 0) + 1
        return counts
    
    def filter_by_task(self, task: str) -> 'Dataset':
        """Create a new dataset with only examples from a specific task."""
        filtered = [ex for ex in self.examples if ex.task == task]
        return Dataset(filtered, name=f"{self.name}_{task}")
    
    def save_jsonl(self, filepath: str) -> None:
        """Save dataset to JSONL file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            for ex in self.examples:
                f.write(json.dumps(ex.to_dict()) + '\n')
        print(f"Saved {len(self)} examples to {filepath}")
    
    @classmethod
    def load_jsonl(cls, filepath: str, name: Optional[str] = None) -> 'Dataset':
        """Load dataset from JSONL file."""
        examples = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(Example.from_dict(data))
        
        if name is None:
            name = Path(filepath).stem
        
        print(f"Loaded {len(examples)} examples from {filepath}")
        return cls(examples, name=name)
    
    def __repr__(self):
        task_counts = self.get_task_counts()
        task_str = ', '.join([f"{t}: {c}" for t, c in task_counts.items()])
        return f"Dataset('{self.name}', {len(self)} examples, tasks=[{task_str}])"


def split_dataset(examples: List[Example], 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  seed: Optional[int] = None) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a list of examples into train/val/test sets.
    
    Args:
        examples: List of Example objects
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total}"
    
    # Shuffle examples
    if seed is not None:
        random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    # Calculate split points
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split
    train_examples = shuffled[:train_end]
    val_examples = shuffled[train_end:val_end]
    test_examples = shuffled[val_end:]
    
    # Create datasets
    train_dataset = Dataset(train_examples, name="train")
    val_dataset = Dataset(val_examples, name="val")
    test_dataset = Dataset(test_examples, name="test")
    
    print(f"Split {n} examples into train={len(train_dataset)}, "
          f"val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def load_task_datasets(data_dir: str, task_name: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load train/val/test datasets for a specific task.
    
    Args:
        data_dir: Base data directory
        task_name: Name of the task
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    task_dir = Path(data_dir) / "tasks" / task_name
    
    train_path = task_dir / "train.jsonl"
    val_path = task_dir / "val.jsonl"
    test_path = task_dir / "test.jsonl"
    
    # Check if files exist
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    
    train_dataset = Dataset.load_jsonl(str(train_path), name=f"{task_name}_train")
    
    # Val and test are optional
    val_dataset = None
    if val_path.exists():
        val_dataset = Dataset.load_jsonl(str(val_path), name=f"{task_name}_val")
    
    test_dataset = None
    if test_path.exists():
        test_dataset = Dataset.load_jsonl(str(test_path), name=f"{task_name}_test")
    
    return train_dataset, val_dataset, test_dataset


def preprocess_dataset(dataset: Dataset, vocab) -> Dataset:
    """
    Preprocess dataset by converting text to word indices.
    
    Args:
        dataset: Dataset to preprocess
        vocab: Vocabulary object with sentence_to_indices method
        
    Returns:
        New dataset with input_indices populated
    """
    preprocessed = []
    for ex in dataset.examples:
        try:
            indices = vocab.sentence_to_indices(ex.input_text)
            preprocessed_ex = Example(
                input_text=ex.input_text,
                task=ex.task,
                label=ex.label,
                input_indices=indices
            )
            preprocessed.append(preprocessed_ex)
        except ValueError as e:
            print(f"Warning: Skipping example due to error: {e}")
            print(f"  Input: {ex.input_text}")
    
    return Dataset(preprocessed, name=dataset.name)


def combine_datasets(datasets: List[Dataset], name: str = "combined") -> Dataset:
    """
    Combine multiple datasets into one.
    
    Args:
        datasets: List of Dataset objects
        name: Name for the combined dataset
        
    Returns:
        Combined dataset
    """
    all_examples = []
    for dataset in datasets:
        all_examples.extend(dataset.examples)
    
    return Dataset(all_examples, name=name)


def create_example(input_text: str, task: str, label: str) -> Example:
    """Helper function to create an Example."""
    return Example(input_text=input_text, task=task, label=label)


# ============================================================================
# Data Generation Helpers
# ============================================================================

def generate_synthetic_examples(vocab_words: List[str], 
                               task_name: str,
                               rules: Dict[str, callable],
                               num_examples: int = 100,
                               seed: Optional[int] = None) -> List[Example]:
    """
    Generate synthetic examples based on rules.
    
    Args:
        vocab_words: List of vocabulary words
        task_name: Name of the task
        rules: Dictionary mapping labels to functions that generate sentences
        num_examples: Number of examples to generate
        seed: Random seed
        
    Returns:
        List of Example objects
    """
    if seed is not None:
        random.seed(seed)
    
    examples = []
    labels = list(rules.keys())
    
    for _ in range(num_examples):
        label = random.choice(labels)
        sentence = rules[label](vocab_words)
        examples.append(Example(input_text=sentence, task=task_name, label=label))
    
    return examples


if __name__ == "__main__":
    # Demo usage
    print("Creating sample dataset...")
    
    # Create some examples
    examples = [
        Example("YOU GO LEFT", "action_mapping", "MOVE_LEFT"),
        Example("I TAKE HERE", "action_mapping", "TAKE_ITEM"),
        Example("YOU GIVE", "action_mapping", "GIVE_ITEM"),
        Example("YES", "intent_classification", "CONFIRMATION"),
        Example("NO", "intent_classification", "NEGATION"),
        Example("YOU GO RIGHT", "action_mapping", "MOVE_RIGHT"),
    ]
    
    dataset = Dataset(examples, name="demo")
    print(dataset)
    print(f"\nTask counts: {dataset.get_task_counts()}")
    print(f"Label counts: {dataset.get_label_counts()}")
    
    # Test splitting
    print("\nSplitting dataset...")
    train, val, test = split_dataset(examples, seed=42)
    print(f"Train: {train}")
    print(f"Val: {val}")
    print(f"Test: {test}")
