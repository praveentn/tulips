"""
Tasks module defining what can be done with the 10-word mini-language.

Each task represents a specific capability that the model should learn:
- Intent classification
- Action mapping
- Response generation
- Direction detection
"""

from enum import Enum
from typing import List, Dict, Set
from dataclasses import dataclass


# ============================================================================
# Task 1: Intent Classification
# ============================================================================

class IntentLabel(Enum):
    """Possible intents in the mini-language."""
    COMMAND = "COMMAND"          # Ordering someone to do something
    STATEMENT = "STATEMENT"      # Stating a fact or action
    CONFIRMATION = "CONFIRMATION"  # Affirming (YES)
    NEGATION = "NEGATION"        # Denying (NO)
    QUERY = "QUERY"              # Asking (implied by structure)


def get_intent_labels() -> List[str]:
    """Get all possible intent labels."""
    return [label.value for label in IntentLabel]


# ============================================================================
# Task 2: Action Mapping
# ============================================================================

class ActionLabel(Enum):
    """Possible actions that can be extracted from sentences."""
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_HERE = "MOVE_HERE"
    GIVE_ITEM = "GIVE_ITEM"
    TAKE_ITEM = "TAKE_ITEM"
    STAY = "STAY"
    NONE = "NONE"


def get_action_labels() -> List[str]:
    """Get all possible action labels."""
    return [label.value for label in ActionLabel]


# ============================================================================
# Task 3: Response Generation
# ============================================================================

class ResponseLabel(Enum):
    """Possible responses to statements/commands."""
    YES = "YES"
    NO = "NO"
    I_GO_LEFT = "I_GO_LEFT"
    I_GO_RIGHT = "I_GO_RIGHT"
    I_TAKE = "I_TAKE"
    I_GIVE = "I_GIVE"
    YOU_TAKE = "YOU_TAKE"
    YOU_GIVE = "YOU_GIVE"
    SILENT = "SILENT"


def get_response_labels() -> List[str]:
    """Get all possible response labels."""
    return [label.value for label in ResponseLabel]


# ============================================================================
# Task 4: Direction Detection
# ============================================================================

class DirectionLabel(Enum):
    """Possible directions mentioned in sentences."""
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    HERE = "HERE"
    NONE = "NONE"


def get_direction_labels() -> List[str]:
    """Get all possible direction labels."""
    return [label.value for label in DirectionLabel]


# ============================================================================
# Task Registry
# ============================================================================

@dataclass
class TaskDefinition:
    """Definition of a task."""
    name: str
    description: str
    labels: List[str]
    
    def __repr__(self):
        return f"Task('{self.name}', {len(self.labels)} labels)"


class TaskRegistry:
    """
    Central registry for all tasks in the mini-language system.
    
    This allows easy addition of new tasks while maintaining consistency.
    """
    
    def __init__(self):
        """Initialize with the base 4 tasks."""
        self.tasks: Dict[str, TaskDefinition] = {}
        self._register_default_tasks()
    
    def _register_default_tasks(self):
        """Register the 4 default tasks."""
        self.register_task(
            "intent_classification",
            "Classify the intent of a sentence",
            get_intent_labels()
        )
        
        self.register_task(
            "action_mapping",
            "Map a sentence to a concrete action",
            get_action_labels()
        )
        
        self.register_task(
            "response_generation",
            "Generate an appropriate response",
            get_response_labels()
        )
        
        self.register_task(
            "direction_detection",
            "Detect which direction is mentioned",
            get_direction_labels()
        )
    
    def register_task(self, name: str, description: str, labels: List[str]) -> None:
        """
        Register a new task.
        
        Args:
            name: Task name (unique identifier)
            description: Human-readable description
            labels: List of possible labels for this task
        """
        if name in self.tasks:
            print(f"Warning: Task '{name}' already registered. Overwriting.")
        
        self.tasks[name] = TaskDefinition(
            name=name,
            description=description,
            labels=labels
        )
        print(f"Registered task: {name} ({len(labels)} labels)")
    
    def get_task(self, name: str) -> TaskDefinition:
        """Get a task definition by name."""
        if name not in self.tasks:
            raise ValueError(f"Task '{name}' not registered. Available: {list(self.tasks.keys())}")
        return self.tasks[name]
    
    def get_all_tasks(self) -> List[TaskDefinition]:
        """Get all registered tasks."""
        return list(self.tasks.values())
    
    def get_task_names(self) -> List[str]:
        """Get names of all registered tasks."""
        return list(self.tasks.keys())
    
    def get_labels_for_task(self, task_name: str) -> List[str]:
        """Get all labels for a specific task."""
        return self.get_task(task_name).labels
    
    def __repr__(self):
        task_list = '\n  '.join([f"{name}: {task.description}" 
                                 for name, task in self.tasks.items()])
        return f"TaskRegistry with {len(self.tasks)} tasks:\n  {task_list}"


def get_default_task_registry() -> TaskRegistry:
    """Get the default task registry with 4 base tasks."""
    return TaskRegistry()


# ============================================================================
# Task-Specific Semantic Rules (for data generation and understanding)
# ============================================================================

def get_intent_rules() -> Dict[str, str]:
    """
    Get rules for intent classification.
    
    Returns:
        Dictionary describing when each intent applies
    """
    return {
        "COMMAND": "Sentence starts with 'YOU' and contains an action verb",
        "STATEMENT": "Sentence starts with 'I' and contains an action verb",
        "CONFIRMATION": "Sentence is just 'YES'",
        "NEGATION": "Sentence is just 'NO'",
        "QUERY": "Sentence structure implies a question (less common in 10-word language)"
    }


def get_action_rules() -> Dict[str, str]:
    """
    Get rules for action mapping.
    
    Returns:
        Dictionary describing when each action applies
    """
    return {
        "MOVE_LEFT": "Contains 'GO LEFT' or 'TAKE LEFT'",
        "MOVE_RIGHT": "Contains 'GO RIGHT' or 'TAKE RIGHT'",
        "MOVE_HERE": "Contains 'GO HERE' or 'TAKE HERE'",
        "GIVE_ITEM": "Contains 'GIVE'",
        "TAKE_ITEM": "Contains 'TAKE' without direction",
        "STAY": "No movement action mentioned",
        "NONE": "No clear action"
    }


def get_response_rules() -> Dict[str, str]:
    """
    Get rules for response generation.
    
    Returns:
        Dictionary describing when each response is appropriate
    """
    return {
        "YES": "Appropriate for confirming commands or statements",
        "NO": "Appropriate for rejecting commands or statements",
        "I_GO_LEFT": "Response to 'YOU GO LEFT' from perspective of 'I'",
        "I_GO_RIGHT": "Response to 'YOU GO RIGHT' from perspective of 'I'",
        "I_TAKE": "Response to 'YOU GIVE' or acknowledgment of taking",
        "I_GIVE": "Response to 'YOU TAKE' or acknowledgment of giving",
        "YOU_TAKE": "Command for other to take",
        "YOU_GIVE": "Command for other to give",
        "SILENT": "No verbal response needed"
    }


# ============================================================================
# Demo and Testing
# ============================================================================

if __name__ == "__main__":
    # Create and display task registry
    registry = get_default_task_registry()
    print(registry)
    print("\n" + "="*60)
    
    # Display each task's labels
    for task_name in registry.get_task_names():
        task = registry.get_task(task_name)
        print(f"\n{task.name}:")
        print(f"  Description: {task.description}")
        print(f"  Labels ({len(task.labels)}): {', '.join(task.labels)}")
    
    # Display semantic rules
    print("\n" + "="*60)
    print("\nIntent Classification Rules:")
    for intent, rule in get_intent_rules().items():
        print(f"  {intent}: {rule}")
    
    print("\nAction Mapping Rules:")
    for action, rule in get_action_rules().items():
        print(f"  {action}: {rule}")
