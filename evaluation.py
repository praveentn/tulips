"""
Evaluation module with comprehensive metrics and analysis.

Provides:
- Accuracy, precision, recall, F1
- Confusion matrices
- Per-class metrics
- Error analysis
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json


class ConfusionMatrix:
    """Confusion matrix for a classification task."""
    
    def __init__(self, labels: List[str]):
        """
        Initialize confusion matrix.
        
        Args:
            labels: List of all possible labels
        """
        self.labels = labels
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        
        # Matrix: rows are true labels, columns are predicted labels
        self.matrix = [[0 for _ in labels] for _ in labels]
    
    def add(self, true_label: str, predicted_label: str) -> None:
        """Add a prediction to the confusion matrix."""
        true_idx = self.label_to_idx[true_label]
        pred_idx = self.label_to_idx[predicted_label]
        self.matrix[true_idx][pred_idx] += 1
    
    def get_accuracy(self) -> float:
        """Calculate overall accuracy."""
        correct = sum(self.matrix[i][i] for i in range(len(self.labels)))
        total = sum(sum(row) for row in self.matrix)
        return correct / total if total > 0 else 0.0
    
    def get_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate precision, recall, and F1 for each class.
        
        Returns:
            Dictionary mapping label -> {precision, recall, f1, support}
        """
        metrics = {}
        
        for i, label in enumerate(self.labels):
            # True positives: diagonal element
            tp = self.matrix[i][i]
            
            # False positives: sum of column i (excluding diagonal)
            fp = sum(self.matrix[j][i] for j in range(len(self.labels)) if j != i)
            
            # False negatives: sum of row i (excluding diagonal)
            fn = sum(self.matrix[i][j] for j in range(len(self.labels)) if j != i)
            
            # Support: total true instances of this class
            support = sum(self.matrix[i])
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support
            }
        
        return metrics
    
    def get_macro_average(self) -> Dict[str, float]:
        """Calculate macro-averaged precision, recall, and F1."""
        per_class = self.get_per_class_metrics()
        
        n = len(per_class)
        avg_precision = sum(m["precision"] for m in per_class.values()) / n
        avg_recall = sum(m["recall"] for m in per_class.values()) / n
        avg_f1 = sum(m["f1"] for m in per_class.values()) / n
        
        return {
            "macro_precision": avg_precision,
            "macro_recall": avg_recall,
            "macro_f1": avg_f1
        }
    
    def print_matrix(self, max_width: int = 10) -> None:
        """Print the confusion matrix in a readable format."""
        # Print header
        header = " " * max_width + " | " + " ".join(f"{label[:8]:>8}" for label in self.labels)
        print(header)
        print("-" * len(header))
        
        # Print rows
        for i, true_label in enumerate(self.labels):
            row_str = f"{true_label[:max_width]:<{max_width}} | "
            row_str += " ".join(f"{self.matrix[i][j]:>8}" for j in range(len(self.labels)))
            print(row_str)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "labels": self.labels,
            "matrix": self.matrix,
            "accuracy": self.get_accuracy(),
            "per_class_metrics": self.get_per_class_metrics(),
            "macro_average": self.get_macro_average()
        }


class Evaluator:
    """Comprehensive evaluator for the model."""
    
    def __init__(self, model, vocab):
        """
        Initialize evaluator.
        
        Args:
            model: AddSubModel instance
            vocab: Vocabulary instance
        """
        self.model = model
        self.vocab = vocab
    
    def evaluate_dataset(self, dataset, task_name: Optional[str] = None) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: Dataset to evaluate
            task_name: If specified, only evaluate on this task
            
        Returns:
            Dictionary with comprehensive metrics
        """
        # Filter dataset by task if specified
        if task_name:
            dataset = dataset.filter_by_task(task_name)
        
        # Get all unique tasks in the dataset
        tasks = set(ex.task for ex in dataset.examples)
        
        results = {}
        
        for task in tasks:
            # Get labels for this task
            labels = self.model.task_labels[task]
            
            # Create confusion matrix
            cm = ConfusionMatrix(labels)
            
            # Collect predictions and errors
            errors = []
            correct_examples = []
            
            for ex in dataset.examples:
                if ex.task != task:
                    continue
                
                if ex.input_indices is None:
                    ex.input_indices = self.vocab.sentence_to_indices(ex.input_text)
                
                # Predict
                predicted = self.model.predict(ex.input_indices, task)
                
                # Update confusion matrix
                cm.add(ex.label, predicted)
                
                # Track errors and correct predictions
                if predicted != ex.label:
                    errors.append({
                        "input": ex.input_text,
                        "true_label": ex.label,
                        "predicted_label": predicted
                    })
                else:
                    correct_examples.append({
                        "input": ex.input_text,
                        "label": ex.label
                    })
            
            # Compile results for this task
            results[task] = {
                "confusion_matrix": cm,
                "accuracy": cm.get_accuracy(),
                "per_class_metrics": cm.get_per_class_metrics(),
                "macro_average": cm.get_macro_average(),
                "num_examples": len([ex for ex in dataset.examples if ex.task == task]),
                "num_errors": len(errors),
                "errors": errors,
                "num_correct": len(correct_examples)
            }
        
        # Calculate overall metrics (across all tasks)
        if len(results) > 1:
            total_correct = sum(r["num_correct"] for r in results.values())
            total_examples = sum(r["num_examples"] for r in results.values())
            overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0
            
            results["overall"] = {
                "accuracy": overall_accuracy,
                "total_examples": total_examples,
                "total_errors": total_examples - total_correct
            }
        
        return results
    
    def print_evaluation_report(self, results: Dict, show_errors: bool = True, max_errors: int = 10) -> None:
        """
        Print a comprehensive evaluation report.
        
        Args:
            results: Results dictionary from evaluate_dataset()
            show_errors: Whether to show example errors
            max_errors: Maximum number of errors to show per task
        """
        print("\n" + "="*70)
        print("EVALUATION REPORT")
        print("="*70)
        
        # Overall metrics (if multi-task)
        if "overall" in results:
            overall = results["overall"]
            print(f"\nOverall Performance:")
            print(f"  Accuracy: {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)")
            print(f"  Total Examples: {overall['total_examples']}")
            print(f"  Total Errors: {overall['total_errors']}")
        
        # Per-task metrics
        for task_name, task_results in results.items():
            if task_name == "overall":
                continue
            
            print(f"\n{'='*70}")
            print(f"Task: {task_name}")
            print(f"{'='*70}")
            
            print(f"\nAccuracy: {task_results['accuracy']:.4f} ({task_results['accuracy']*100:.2f}%)")
            print(f"Examples: {task_results['num_examples']}")
            print(f"Errors: {task_results['num_errors']}")
            
            # Macro-averaged metrics
            macro = task_results['macro_average']
            print(f"\nMacro-Averaged Metrics:")
            print(f"  Precision: {macro['macro_precision']:.4f}")
            print(f"  Recall: {macro['macro_recall']:.4f}")
            print(f"  F1 Score: {macro['macro_f1']:.4f}")
            
            # Per-class metrics
            print(f"\nPer-Class Metrics:")
            per_class = task_results['per_class_metrics']
            for label, metrics in per_class.items():
                print(f"  {label}:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1 Score:  {metrics['f1']:.4f}")
                print(f"    Support:   {metrics['support']}")
            
            # Confusion matrix
            print(f"\nConfusion Matrix:")
            task_results['confusion_matrix'].print_matrix()
            
            # Show errors
            if show_errors and task_results['errors']:
                print(f"\nSample Errors (showing up to {max_errors}):")
                for i, error in enumerate(task_results['errors'][:max_errors]):
                    print(f"  {i+1}. Input: '{error['input']}'")
                    print(f"     True: {error['true_label']}, Predicted: {error['predicted_label']}")
                
                if len(task_results['errors']) > max_errors:
                    remaining = len(task_results['errors']) - max_errors
                    print(f"  ... and {remaining} more errors")
    
    def analyze_feature_importance(self, task_name: str, top_k: int = 20) -> List[Tuple]:
        """
        Analyze which features are most important for a task.
        
        Args:
            task_name: Name of the task
            top_k: Number of top features to return
            
        Returns:
            List of (feature, importance) tuples
        """
        return self.model.get_feature_importance(task_name, top_k)
    
    def print_feature_importance(self, task_name: str, top_k: int = 20) -> None:
        """Print feature importance for a task."""
        features = self.analyze_feature_importance(task_name, top_k)
        
        print(f"\nTop {top_k} Most Important Features for '{task_name}':")
        print("-" * 60)
        
        for i, (feature, importance) in enumerate(features, 1):
            # Format feature for display
            if feature[0] == 'unigram':
                word = self.vocab.get_word(feature[1])
                feature_str = f"unigram: '{word}'"
            elif feature[0] == 'bigram':
                word1 = self.vocab.get_word(feature[1])
                word2 = self.vocab.get_word(feature[2])
                feature_str = f"bigram: '{word1} {word2}'"
            else:
                feature_str = str(feature)
            
            print(f"  {i:2d}. {feature_str:<40} importance: {importance}")


if __name__ == "__main__":
    # Demo usage
    from data_utils import Example, Dataset
    from model import AddSubModel
    from vocab import get_default_vocab
    
    print("Setting up evaluation demo...")
    
    # Create vocab and model
    vocab = get_default_vocab()
    task_labels = {
        "test_task": ["CLASS_A", "CLASS_B", "CLASS_C"]
    }
    model = AddSubModel(vocab_size=len(vocab), task_labels=task_labels)
    
    # Create and train on some mock data
    examples = [
        Example("YOU GO LEFT", "test_task", "CLASS_A"),
        Example("I TAKE HERE", "test_task", "CLASS_B"),
        Example("YES", "test_task", "CLASS_C"),
        Example("YOU GO RIGHT", "test_task", "CLASS_A"),
        Example("NO", "test_task", "CLASS_C"),
    ]
    
    for ex in examples:
        ex.input_indices = vocab.sentence_to_indices(ex.input_text)
        model.train_on_example(ex.input_indices, ex.task, ex.label)
    
    # Evaluate
    dataset = Dataset(examples, name="test")
    evaluator = Evaluator(model, vocab)
    results = evaluator.evaluate_dataset(dataset)
    
    # Print report
    evaluator.print_evaluation_report(results)
    evaluator.print_feature_importance("test_task", top_k=10)
