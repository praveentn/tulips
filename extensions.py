"""
Extensions module for adding new words and new tasks to the system.

Supports:
- Adding a new word to the vocabulary
- Adding a new task with labels
- Reinitializing model parameters accordingly
- Generating new training data (if needed)
"""

from typing import List, Dict, Tuple, Optional
import random


class LanguageExtender:
    """
    Handles extension of the mini-language with new words or tasks.
    
    This enables the extensibility requirement: after initial training,
    we can add new words or tasks and retrain.
    """
    
    def __init__(self, vocab, model, task_registry):
        """
        Initialize extender.
        
        Args:
            vocab: Vocabulary instance
            model: AddSubModel instance
            task_registry: TaskRegistry instance
        """
        self.vocab = vocab
        self.model = model
        self.task_registry = task_registry
    
    def add_new_word(self, word: str) -> None:
        """
        Add a new word to the vocabulary.
        
        This automatically:
        1. Adds word to vocabulary
        2. Updates model's vocab_size
        3. Initializes parameters for the new word (all zeros initially)
        
        Args:
            word: New word to add (will be uppercased)
        """
        word = word.upper()
        
        # Check if word already exists
        if self.vocab.contains(word):
            print(f"Word '{word}' already in vocabulary")
            return
        
        print(f"\nAdding new word: '{word}'")
        print("-" * 50)
        
        # Add to vocabulary
        old_vocab_size = len(self.vocab)
        self.vocab.add_word(word)
        new_vocab_size = len(self.vocab)
        
        # Update model's vocab_size
        self.model.vocab_size = new_vocab_size
        
        print(f"✓ Vocabulary updated: {old_vocab_size} -> {new_vocab_size} words")
        print(f"✓ Model parameters initialized for new word")
        print(f"  (All scores for new word start at 0)")
        print(f"\nNext steps:")
        print(f"  1. Generate or provide training examples using '{word}'")
        print(f"  2. Retrain the model to learn patterns with this word")
    
    def add_new_task(self, task_name: str, description: str, labels: List[str]) -> None:
        """
        Add a new task to the system.
        
        This automatically:
        1. Registers task in the task registry
        2. Adds task to model (initializes parameter table)
        3. Updates model's task_labels
        
        Args:
            task_name: Name of the new task (unique identifier)
            description: Human-readable description
            labels: List of possible labels for this task
        """
        print(f"\nAdding new task: '{task_name}'")
        print("-" * 50)
        
        # Register in task registry
        self.task_registry.register_task(task_name, description, labels)
        
        # Add to model
        self.model.add_task(task_name, labels)
        
        print(f"✓ Task registered with {len(labels)} labels")
        print(f"✓ Model parameters initialized (all zeros)")
        print(f"\nNext steps:")
        print(f"  1. Generate or provide training data for '{task_name}'")
        print(f"  2. Train the model on this task")
    
    def generate_extended_examples(self, 
                                   new_word: str,
                                   task_name: str,
                                   num_examples: int = 10) -> List[Dict]:
        """
        Generate example sentences using a new word for a specific task.
        
        This is a helper for creating synthetic data when a new word is added.
        The examples are simple combinations based on patterns.
        
        Args:
            new_word: The newly added word
            task_name: Task to generate examples for
            num_examples: Number of examples to generate
            
        Returns:
            List of example dictionaries
        """
        examples = []
        
        # Get existing words for combination
        existing_words = [w for w in self.vocab.words if w != new_word.upper()]
        
        # Get labels for this task
        if task_name not in self.model.task_labels:
            print(f"Warning: Task '{task_name}' not found in model")
            return examples
        
        labels = self.model.task_labels[task_name]
        
        print(f"\nGenerating {num_examples} examples with '{new_word}' for '{task_name}'...")
        
        # Generate random combinations
        for i in range(num_examples):
            # Create a sentence with the new word
            num_words = random.randint(1, 3)
            words = [new_word.upper()]
            
            # Add some existing words
            for _ in range(num_words - 1):
                words.append(random.choice(existing_words))
            
            random.shuffle(words)
            sentence = " ".join(words)
            
            # Assign a random label (in real scenario, use rules)
            label = random.choice(labels)
            
            examples.append({
                "input": sentence,
                "task": task_name,
                "label": label
            })
        
        print(f"✓ Generated {len(examples)} examples")
        print(f"  Note: These are random combinations. In practice, you should")
        print(f"  generate examples based on semantic rules for better quality.")
        
        return examples
    
    def get_extension_summary(self) -> Dict:
        """
        Get a summary of all extensions made.
        
        Returns:
            Dictionary with extension information
        """
        original_vocab_size = 10  # The original 10-word vocabulary
        current_vocab_size = len(self.vocab)
        added_words = current_vocab_size - original_vocab_size
        
        original_tasks = 4  # The original 4 tasks
        current_tasks = len(self.model.task_labels)
        added_tasks = current_tasks - original_tasks
        
        return {
            "original_vocab_size": original_vocab_size,
            "current_vocab_size": current_vocab_size,
            "added_words": added_words,
            "new_words": self.vocab.words[10:] if added_words > 0 else [],
            "original_tasks": original_tasks,
            "current_tasks": current_tasks,
            "added_tasks": added_tasks,
            "all_tasks": list(self.model.task_labels.keys())
        }
    
    def print_extension_summary(self) -> None:
        """Print a formatted summary of extensions."""
        summary = self.get_extension_summary()
        
        print("\n" + "="*60)
        print("EXTENSION SUMMARY")
        print("="*60)
        
        print(f"\nVocabulary:")
        print(f"  Original size: {summary['original_vocab_size']} words")
        print(f"  Current size:  {summary['current_vocab_size']} words")
        print(f"  Added:         {summary['added_words']} words")
        if summary['new_words']:
            print(f"  New words:     {', '.join(summary['new_words'])}")
        
        print(f"\nTasks:")
        print(f"  Original:      {summary['original_tasks']} tasks")
        print(f"  Current:       {summary['current_tasks']} tasks")
        print(f"  Added:         {summary['added_tasks']} tasks")
        print(f"  All tasks:     {', '.join(summary['all_tasks'])}")


def demonstrate_extension_workflow(vocab, model, task_registry):
    """
    Demonstrate the complete workflow for extending the language.
    
    Args:
        vocab: Vocabulary instance
        model: AddSubModel instance
        task_registry: TaskRegistry instance
    """
    extender = LanguageExtender(vocab, model, task_registry)
    
    print("\n" + "="*70)
    print("LANGUAGE EXTENSION DEMONSTRATION")
    print("="*70)
    
    print("\nScenario: Adding a new word 'UP' and a new task")
    
    # Step 1: Add new word
    extender.add_new_word("UP")
    
    # Step 2: Add new task
    extender.add_new_task(
        task_name="movement_type",
        description="Classify the type of movement",
        labels=["HORIZONTAL", "VERTICAL", "STATIC"]
    )
    
    # Step 3: Show summary
    extender.print_extension_summary()
    
    print("\n" + "="*70)
    print("Extension workflow complete!")
    print("="*70)


if __name__ == "__main__":
    from vocab import get_default_vocab
    from model import AddSubModel
    from tasks import get_default_task_registry
    
    print("Setting up extension demo...")
    
    # Create initial system
    vocab = get_default_vocab()
    task_registry = get_default_task_registry()
    
    task_labels = {name: task.labels for name, task in task_registry.tasks.items()}
    model = AddSubModel(vocab_size=len(vocab), task_labels=task_labels)
    
    # Demonstrate extension
    demonstrate_extension_workflow(vocab, model, task_registry)
