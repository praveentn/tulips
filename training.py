"""
Training module with learning loop, logging, and metrics tracking.

Supports:
- Multi-epoch training
- Training and validation metrics per epoch
- Learning curves tracking
- Early stopping (optional)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    train_accuracy: float
    train_loss: float  # Number of errors
    val_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    num_updates: int = 0
    time_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "train_accuracy": self.train_accuracy,
            "train_loss": self.train_loss,
            "val_accuracy": self.val_accuracy,
            "val_loss": self.val_loss,
            "num_updates": self.num_updates,
            "time_seconds": self.time_seconds
        }


@dataclass
class TrainingHistory:
    """Container for training history across epochs."""
    metrics: List[EpochMetrics] = field(default_factory=list)
    
    def add_epoch(self, metrics: EpochMetrics) -> None:
        """Add metrics for an epoch."""
        self.metrics.append(metrics)
    
    def get_train_accuracies(self) -> List[float]:
        """Get list of training accuracies."""
        return [m.train_accuracy for m in self.metrics]
    
    def get_val_accuracies(self) -> List[float]:
        """Get list of validation accuracies."""
        return [m.val_accuracy for m in self.metrics if m.val_accuracy is not None]
    
    def get_train_losses(self) -> List[float]:
        """Get list of training losses."""
        return [m.train_loss for m in self.metrics]
    
    def get_val_losses(self) -> List[float]:
        """Get list of validation losses."""
        return [m.val_loss for m in self.metrics if m.val_loss is not None]
    
    def get_best_val_accuracy(self) -> Tuple[int, float]:
        """Get epoch and value of best validation accuracy."""
        val_accs = [(m.epoch, m.val_accuracy) for m in self.metrics 
                    if m.val_accuracy is not None]
        if not val_accs:
            return -1, 0.0
        return max(val_accs, key=lambda x: x[1])
    
    def __len__(self) -> int:
        """Return number of epochs tracked."""
        return len(self.metrics)
    
    def __repr__(self):
        if not self.metrics:
            return "TrainingHistory(empty)"
        last = self.metrics[-1]
        return (f"TrainingHistory({len(self.metrics)} epochs, "
                f"last train_acc={last.train_accuracy:.3f}, "
                f"last val_acc={last.val_accuracy:.3f if last.val_accuracy else 'N/A'})")


class Trainer:
    """
    Trainer for the AddSubModel.
    
    Handles:
    - Training loop over multiple epochs
    - Evaluation on train and validation sets
    - Metrics tracking and history
    - Progress reporting
    """
    
    def __init__(self, model, vocab, verbose: bool = True):
        """
        Initialize trainer.
        
        Args:
            model: AddSubModel instance
            vocab: Vocabulary instance
            verbose: Whether to print progress
        """
        self.model = model
        self.vocab = vocab
        self.verbose = verbose
        self.history = TrainingHistory()
    
    def train_epoch(self, train_dataset, shuffle: bool = True) -> Tuple[float, float, int]:
        """
        Train for one epoch on the training dataset.
        
        Args:
            train_dataset: Dataset object with training examples
            shuffle: Whether to shuffle examples each epoch
            
        Returns:
            Tuple of (accuracy, loss, num_updates)
            - accuracy: fraction of correct predictions
            - loss: number of incorrect predictions
            - num_updates: number of parameter updates made
        """
        if shuffle:
            train_dataset.shuffle()
        
        correct = 0
        total = 0
        num_updates = 0
        
        for example in train_dataset.examples:
            if example.input_indices is None:
                # Preprocess on the fly if needed
                example.input_indices = self.vocab.sentence_to_indices(example.input_text)
            
            # Train on this example
            predicted, is_correct, updates = self.model.train_on_example(
                example.input_indices,
                example.task,
                example.label
            )
            
            if is_correct:
                correct += 1
            total += 1
            num_updates += updates
        
        accuracy = correct / total if total > 0 else 0.0
        loss = total - correct  # Number of errors
        
        return accuracy, loss, num_updates
    
    def evaluate(self, dataset) -> Tuple[float, float, Dict[str, float]]:
        """
        Evaluate model on a dataset (no training).
        
        Args:
            dataset: Dataset object
            
        Returns:
            Tuple of (accuracy, loss, per_task_accuracy)
            - accuracy: overall accuracy
            - loss: number of errors
            - per_task_accuracy: dictionary of task -> accuracy
        """
        correct = 0
        total = 0
        
        # Track per-task performance
        task_correct = {}
        task_total = {}
        
        for example in dataset.examples:
            if example.input_indices is None:
                example.input_indices = self.vocab.sentence_to_indices(example.input_text)
            
            # Predict only (no training)
            predicted = self.model.predict(example.input_indices, example.task)
            
            is_correct = (predicted == example.label)
            if is_correct:
                correct += 1
                task_correct[example.task] = task_correct.get(example.task, 0) + 1
            
            total += 1
            task_total[example.task] = task_total.get(example.task, 0) + 1
        
        accuracy = correct / total if total > 0 else 0.0
        loss = total - correct
        
        # Calculate per-task accuracy
        per_task_accuracy = {
            task: task_correct.get(task, 0) / task_total[task]
            for task in task_total
        }
        
        return accuracy, loss, per_task_accuracy
    
    def train(self, 
              train_dataset,
              val_dataset=None,
              num_epochs: int = 10,
              shuffle: bool = True,
              early_stopping_patience: Optional[int] = None) -> TrainingHistory:
        """
        Train the model for multiple epochs.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            num_epochs: Number of epochs to train
            shuffle: Whether to shuffle training data each epoch
            early_stopping_patience: Stop if val accuracy doesn't improve for N epochs
            
        Returns:
            TrainingHistory object with metrics for each epoch
        """
        if self.verbose:
            print(f"\nStarting training for {num_epochs} epochs...")
            print(f"Train dataset: {len(train_dataset)} examples")
            if val_dataset:
                print(f"Val dataset: {len(val_dataset)} examples")
            print("="*60)
        
        best_val_accuracy = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train for one epoch
            train_acc, train_loss, num_updates = self.train_epoch(train_dataset, shuffle)
            
            # Evaluate on validation set if provided
            val_acc, val_loss = None, None
            if val_dataset:
                val_acc, val_loss, _ = self.evaluate(val_dataset)
                
                # Early stopping check
                if early_stopping_patience:
                    if val_acc > best_val_accuracy:
                        best_val_accuracy = val_acc
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
            
            epoch_time = time.time() - epoch_start
            
            # Create metrics for this epoch
            metrics = EpochMetrics(
                epoch=epoch,
                train_accuracy=train_acc,
                train_loss=train_loss,
                val_accuracy=val_acc,
                val_loss=val_loss,
                num_updates=num_updates,
                time_seconds=epoch_time
            )
            
            self.history.add_epoch(metrics)
            
            # Print progress
            if self.verbose:
                val_str = f", val_acc={val_acc:.4f}" if val_acc is not None else ""
                print(f"Epoch {epoch:3d}/{num_epochs}: "
                      f"train_acc={train_acc:.4f}, train_loss={int(train_loss)}"
                      f"{val_str}, updates={num_updates}, time={epoch_time:.2f}s")
            
            # Early stopping
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no improvement for {early_stopping_patience} epochs)")
                break
        
        if self.verbose:
            print("="*60)
            print(f"Training complete! {len(self.history)} epochs")
            if val_dataset:
                best_epoch, best_acc = self.history.get_best_val_accuracy()
                print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
        
        return self.history
    
    def get_history(self) -> TrainingHistory:
        """Get the training history."""
        return self.history


def print_evaluation_summary(dataset_name: str, 
                            accuracy: float, 
                            loss: float,
                            per_task_accuracy: Dict[str, float]) -> None:
    """
    Print a formatted evaluation summary.
    
    Args:
        dataset_name: Name of the dataset (e.g., "Test")
        accuracy: Overall accuracy
        loss: Number of errors
        per_task_accuracy: Dictionary of task -> accuracy
    """
    print(f"\n{dataset_name} Set Evaluation:")
    print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Number of Errors: {int(loss)}")
    print(f"  Per-Task Accuracy:")
    for task, acc in per_task_accuracy.items():
        print(f"    {task}: {acc:.4f} ({acc*100:.2f}%)")


if __name__ == "__main__":
    # Demo usage with mock data
    from data_utils import Example, Dataset
    from model import AddSubModel
    from vocab import get_default_vocab
    
    print("Setting up mock training demo...")
    
    # Create vocab and model
    vocab = get_default_vocab()
    task_labels = {
        "mock_task": ["LABEL_A", "LABEL_B", "LABEL_C"]
    }
    model = AddSubModel(vocab_size=len(vocab), task_labels=task_labels)
    
    # Create mock datasets
    train_examples = [
        Example("YOU GO LEFT", "mock_task", "LABEL_A"),
        Example("I TAKE HERE", "mock_task", "LABEL_B"),
        Example("YES", "mock_task", "LABEL_C"),
        Example("YOU GO RIGHT", "mock_task", "LABEL_A"),
        Example("NO", "mock_task", "LABEL_C"),
    ]
    
    val_examples = [
        Example("YOU GO LEFT", "mock_task", "LABEL_A"),
        Example("NO", "mock_task", "LABEL_C"),
    ]
    
    # Preprocess
    for ex in train_examples + val_examples:
        ex.input_indices = vocab.sentence_to_indices(ex.input_text)
    
    train_dataset = Dataset(train_examples, name="mock_train")
    val_dataset = Dataset(val_examples, name="mock_val")
    
    # Train
    trainer = Trainer(model, vocab, verbose=True)
    history = trainer.train(train_dataset, val_dataset, num_epochs=5)
    
    print(f"\nFinal history: {history}")
