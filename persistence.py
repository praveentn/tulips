"""
Persistence module for saving and loading models, vocabularies, and training state.
"""

import json
from pathlib import Path
from typing import Dict, Any
import pickle


def save_model_checkpoint(model, vocab, history, filepath: str) -> None:
    """
    Save a complete checkpoint including model, vocabulary, and training history.
    
    Args:
        model: AddSubModel instance
        vocab: Vocabulary instance
        history: TrainingHistory instance
        filepath: Path to save checkpoint
    """
    checkpoint_dir = Path(filepath).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle for easy loading
    checkpoint = {
        "model": model,
        "vocab": vocab,
        "history": history
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved to {filepath}")


def load_model_checkpoint(filepath: str) -> Dict[str, Any]:
    """
    Load a complete checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        
    Returns:
        Dictionary with keys: "model", "vocab", "history"
    """
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def save_training_config(config: Dict, filepath: str) -> None:
    """
    Save training configuration to JSON.
    
    Args:
        config: Dictionary with training configuration
        filepath: Path to save config
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved to {filepath}")


def load_training_config(filepath: str) -> Dict:
    """Load training configuration from JSON."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    print(f"Config loaded from {filepath}")
    return config


def export_model_parameters(model, filepath: str) -> None:
    """
    Export model parameters in human-readable JSON format.
    
    Args:
        model: AddSubModel instance
        filepath: Path to save parameters
    """
    # Convert parameters to serializable format
    params_export = {}
    
    for task_name, task_params in model.params.items():
        params_export[task_name] = {}
        for feature, label_scores in task_params.items():
            feature_str = str(feature)
            params_export[task_name][feature_str] = dict(label_scores)
    
    export_data = {
        "vocab_size": model.vocab_size,
        "task_labels": model.task_labels,
        "use_bigrams": model.use_bigrams,
        "parameters": params_export,
        "statistics": model.get_stats()
    }
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Model parameters exported to {filepath}")


def export_history_to_json(history, filepath: str) -> None:
    """
    Export training history to JSON format.
    
    Args:
        history: TrainingHistory instance
        filepath: Path to save history
    """
    history_data = {
        "num_epochs": len(history),
        "metrics": [m.to_dict() for m in history.metrics]
    }
    
    with open(filepath, 'w') as f:
        json.dump(history_data, f, indent=2)
    
    print(f"Training history exported to {filepath}")


if __name__ == "__main__":
    print("Persistence module - used for saving/loading models and training state")
