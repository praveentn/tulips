"""
Mini-Language Learning System

A machine learning system that learns a 10-word language using only
addition and subtraction operations.
"""

__version__ = "1.0.0"

from .vocab import Vocabulary, get_default_vocab
from .tasks import TaskRegistry, get_default_task_registry
from .model import AddSubModel
from .data_utils import Dataset, Example
from .training import Trainer, TrainingHistory
from .evaluation import Evaluator
from .extensions import LanguageExtender
from .persistence import save_model_checkpoint, load_model_checkpoint

__all__ = [
    'Vocabulary',
    'get_default_vocab',
    'TaskRegistry',
    'get_default_task_registry',
    'AddSubModel',
    'Dataset',
    'Example',
    'Trainer',
    'TrainingHistory',
    'Evaluator',
    'LanguageExtender',
    'save_model_checkpoint',
    'load_model_checkpoint',
]
