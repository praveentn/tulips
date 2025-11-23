"""
Core model using ONLY addition and subtraction operations.

This model learns through:
- Feature extraction (unigrams and bigrams)
- Integer-valued parameter tables (scores)
- Prediction via summation (add only) and argmax
- Learning via Â±1 updates (perceptron-like)

NO matrix multiplication, NO dot products, NO standard neural network operations.
"""

from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import json


def _default_label_dict():
    """Factory function for nested defaultdict - needed for pickle support."""
    return defaultdict(int)


class AddSubModel:
    """
    A learning model that uses ONLY addition and subtraction.
    
    Architecture:
    - Features: unigrams (single words) and bigrams (word pairs)
    - Parameters: score[feature][label] = integer count
    - Prediction: For each label, sum all relevant scores -> pick argmax
    - Learning: On errors, +1 to correct label features, -1 to predicted label features
    
    This is similar to a perceptron but explicitly uses only add/sub operations.
    """
    
    def __init__(self, vocab_size: int, task_labels: Dict[str, List[str]], use_bigrams: bool = True):
        """
        Initialize the model.
        
        Args:
            vocab_size: Number of words in vocabulary
            task_labels: Dictionary mapping task names to lists of labels
            use_bigrams: Whether to use bigram features (in addition to unigrams)
        """
        self.vocab_size = vocab_size
        self.task_labels = task_labels
        self.use_bigrams = use_bigrams
        
        # Parameter tables: nested dictionaries
        # Structure: params[task][feature] = {label1: score1, label2: score2, ...}
        self.params: Dict[str, Dict] = {}
        
        # Initialize parameters for each task
        for task_name, labels in task_labels.items():
            self.params[task_name] = defaultdict(_default_label_dict)
        
        # Statistics for tracking
        self.update_count = 0
        self.prediction_count = 0
    
    def _extract_features(self, word_indices: List[int]) -> List[Tuple]:
        """
        Extract features from a sentence (represented as word indices).
        
        Features include:
        - Unigrams: ('unigram', word_idx)
        - Bigrams: ('bigram', word_idx1, word_idx2) if use_bigrams=True
        
        Args:
            word_indices: List of word indices
            
        Returns:
            List of feature tuples
        """
        features = []
        
        # Unigram features
        for idx in word_indices:
            features.append(('unigram', idx))
        
        # Bigram features
        if self.use_bigrams and len(word_indices) >= 2:
            for i in range(len(word_indices) - 1):
                features.append(('bigram', word_indices[i], word_indices[i+1]))
        
        return features
    
    def predict(self, word_indices: List[int], task_name: str) -> str:
        """
        Predict the label for a sentence on a specific task.
        
        Prediction algorithm (ADDITION ONLY):
        1. Extract features from input
        2. For each possible label:
            - Initialize score to 0
            - For each feature, ADD the parameter value for (feature, label)
            - Keep running sum
        3. Return label with maximum score (argmax)
        
        Args:
            word_indices: List of word indices representing the sentence
            task_name: Name of the task
            
        Returns:
            Predicted label (string)
        """
        if task_name not in self.params:
            raise ValueError(f"Task '{task_name}' not in model")
        
        self.prediction_count += 1
        
        # Extract features
        features = self._extract_features(word_indices)
        
        # Get task-specific parameters and labels
        task_params = self.params[task_name]
        possible_labels = self.task_labels[task_name]
        
        # Compute score for each label using ONLY ADDITION
        label_scores = {}
        for label in possible_labels:
            score = 0  # Start with 0
            
            # ADD score for each feature (this is the only operation!)
            for feature in features:
                # Get the score for this (feature, label) pair
                # defaultdict returns 0 if not found
                score = score + task_params[feature][label]  # ADDITION ONLY
            
            label_scores[label] = score
        
        # Find label with maximum score (argmax)
        # Use max() with key function (comparison-based, no arithmetic)
        predicted_label = max(label_scores.keys(), key=lambda lbl: label_scores[lbl])
        
        return predicted_label
    
    def update(self, word_indices: List[int], task_name: str, 
               true_label: str, predicted_label: str) -> int:
        """
        Update model parameters based on prediction error.
        
        Update rule (Â±1 ONLY):
        - If prediction is wrong:
            - For each feature in the input:
                - INCREMENT (+1) the score for (feature, true_label)
                - DECREMENT (-1) the score for (feature, predicted_label)
        
        This is a perceptron-like update using only addition and subtraction.
        
        Args:
            word_indices: List of word indices
            task_name: Name of the task
            true_label: Correct label
            predicted_label: Model's prediction
            
        Returns:
            Number of parameter updates made (0 if correct, 2*num_features if wrong)
        """
        if task_name not in self.params:
            raise ValueError(f"Task '{task_name}' not in model")
        
        # If prediction is correct, no update needed
        if predicted_label == true_label:
            return 0
        
        # Extract features
        features = self._extract_features(word_indices)
        
        # Get task-specific parameters
        task_params = self.params[task_name]
        
        # Update each feature (Â±1 operations only)
        updates_made = 0
        for feature in features:
            # Increase score for correct label
            task_params[feature][true_label] = task_params[feature][true_label] + 1  # +1
            updates_made += 1
            
            # Decrease score for incorrect predicted label
            task_params[feature][predicted_label] = task_params[feature][predicted_label] - 1  # -1
            updates_made += 1
        
        self.update_count += 1
        return updates_made
    
    def train_on_example(self, word_indices: List[int], task_name: str, 
                        true_label: str) -> Tuple[str, bool, int]:
        """
        Train on a single example: predict then update.
        
        Args:
            word_indices: List of word indices
            task_name: Name of the task
            true_label: Correct label
            
        Returns:
            Tuple of (predicted_label, is_correct, num_updates)
        """
        # Predict
        predicted_label = self.predict(word_indices, task_name)
        
        # Update
        num_updates = self.update(word_indices, task_name, true_label, predicted_label)
        
        is_correct = (predicted_label == true_label)
        
        return predicted_label, is_correct, num_updates
    
    def get_feature_importance(self, task_name: str, top_k: int = 20) -> List[Tuple]:
        """
        Get most important features for a task (for interpretability).
        
        Importance = sum of absolute values of all label scores for a feature.
        
        Args:
            task_name: Name of the task
            top_k: Number of top features to return
            
        Returns:
            List of (feature, importance_score) tuples
        """
        if task_name not in self.params:
            raise ValueError(f"Task '{task_name}' not in model")
        
        task_params = self.params[task_name]
        feature_importance = {}
        
        for feature, label_scores in task_params.items():
            # Importance = sum of absolute values (using abs and addition only)
            importance = 0
            for label, score in label_scores.items():
                abs_score = score if score >= 0 else -score  # abs without built-in
                importance = importance + abs_score  # Addition only
            feature_importance[feature] = importance
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), 
                                key=lambda x: x[1], 
                                reverse=True)
        
        return sorted_features[:top_k]
    
    def add_task(self, task_name: str, labels: List[str]) -> None:
        """
        Add a new task to the model (for extensibility).
        
        Args:
            task_name: Name of the new task
            labels: List of possible labels for this task
        """
        if task_name in self.params:
            print(f"Warning: Task '{task_name}' already exists")
            return
        
        self.task_labels[task_name] = labels
        self.params[task_name] = defaultdict(_default_label_dict)
        print(f"Added new task: {task_name} with {len(labels)} labels")
    
    def get_stats(self) -> Dict:
        """Get model statistics."""
        total_params = 0
        non_zero_params = 0
        
        for task_name, task_params in self.params.items():
            for feature, label_scores in task_params.items():
                for label, score in label_scores.items():
                    total_params += 1
                    if score != 0:
                        non_zero_params += 1
        
        return {
            "total_parameters": total_params,
            "non_zero_parameters": non_zero_params,
            "sparsity": 1.0 - (non_zero_params / total_params if total_params > 0 else 0),
            "update_count": self.update_count,
            "prediction_count": self.prediction_count,
            "num_tasks": len(self.params)
        }
    
    def save(self, filepath: str) -> None:
        """
        Save model parameters to JSON file.
        
        Note: defaultdict needs special handling for JSON serialization.
        """
        # Convert nested defaultdicts to regular dicts for JSON
        params_serializable = {}
        for task_name, task_params in self.params.items():
            params_serializable[task_name] = {}
            for feature, label_scores in task_params.items():
                # Convert feature tuple to string key
                feature_key = str(feature)
                params_serializable[task_name][feature_key] = dict(label_scores)
        
        model_dict = {
            "vocab_size": self.vocab_size,
            "task_labels": self.task_labels,
            "use_bigrams": self.use_bigrams,
            "params": params_serializable,
            "stats": self.get_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_dict, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'AddSubModel':
        """Load model from JSON file."""
        with open(filepath, 'r') as f:
            model_dict = json.load(f)
        
        # Create model
        model = cls(
            vocab_size=model_dict["vocab_size"],
            task_labels=model_dict["task_labels"],
            use_bigrams=model_dict["use_bigrams"]
        )
        
        # Load parameters
        for task_name, task_params_dict in model_dict["params"].items():
            for feature_str, label_scores in task_params_dict.items():
                # Convert string key back to tuple
                feature = eval(feature_str)
                for label, score in label_scores.items():
                    model.params[task_name][feature][label] = score
        
        print(f"Model loaded from {filepath}")
        return model
    
    def __repr__(self):
        stats = self.get_stats()
        return (f"AddSubModel(vocab={self.vocab_size}, tasks={stats['num_tasks']}, "
                f"params={stats['non_zero_parameters']}/{stats['total_parameters']}, "
                f"updates={stats['update_count']})")


if __name__ == "__main__":
    # Demo usage
    print("Creating a toy model...")
    
    # Define a simple task
    task_labels = {
        "sentiment": ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    }
    
    model = AddSubModel(vocab_size=10, task_labels=task_labels, use_bigrams=True)
    print(model)
    
    # Simulate some training examples (using word indices)
    examples = [
        ([0, 2, 5], "sentiment", "POSITIVE"),
        ([1, 3, 6], "sentiment", "NEGATIVE"),
        ([0, 2, 5], "sentiment", "POSITIVE"),  # Same as first
    ]
    
    print("\nTraining on examples:")
    for word_indices, task, label in examples:
        pred, correct, updates = model.train_on_example(word_indices, task, label)
        status = "âœ“" if correct else "âœ—"
        print(f"  {status} Input: {word_indices}, True: {label}, Pred: {pred}, Updates: {updates}")
    
    print(f"\n{model}")
    print(f"\nStats: {model.get_stats()}")