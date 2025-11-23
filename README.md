# Mini-Language Learning System

A complete machine learning pipeline that learns a 10-word language using **only addition and subtraction** operations.

## 🎯 Project Overview

This project demonstrates that meaningful machine learning is possible with extremely simple operations - no matrix multiplication, no dot products, no standard neural network operations. Just counting, searching, and ±1 parameter updates.

### Key Features

- ✅ **10-word vocabulary**: I, YOU, GO, GIVE, TAKE, LEFT, RIGHT, HERE, YES, NO
- ✅ **4 learning tasks**: Intent classification, action mapping, response generation, direction detection
- ✅ **Pure add/sub learning**: No matrix multiplication or fancy linear algebra
- ✅ **Fully interpretable**: All parameters are simple integer counts
- ✅ **Extensible**: Can add new words and new tasks dynamically
- ✅ **Complete ML pipeline**: Training, validation, evaluation, metrics, visualization

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
cd mini_language_model

# Install dependencies
pip install -r requirements.txt

# Generate datasets (already done, but you can regenerate)
python src/generate_data.py
```

### Running the Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/mini_language_experiments.ipynb
# Run all cells to see the complete demo
```

## 📁 Project Structure

```
mini_language_model/
├── data/
│   └── tasks/                  # Training/val/test datasets
│       ├── intent_classification/
│       ├── action_mapping/
│       ├── response_generation/
│       └── direction_detection/
├── src/
│   ├── vocab.py               # 10-word vocabulary management
│   ├── tasks.py               # Task definitions and labels
│   ├── model.py               # Add/sub-only learning model
│   ├── data_utils.py          # Dataset loading and processing
│   ├── training.py            # Training loop and metrics
│   ├── evaluation.py          # Evaluation and analysis
│   ├── persistence.py         # Save/load models
│   ├── extensions.py          # Add new words/tasks
│   └── generate_data.py       # Synthetic data generation
├── notebooks/
│   └── mini_language_experiments.ipynb  # Main demo notebook
├── models/                    # Saved models and checkpoints
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## 🧠 How It Works

### The Learning Algorithm

The model uses a **perceptron-like** algorithm with strict constraints:

#### Prediction (Addition Only)
```
For each possible label:
    score = 0
    For each feature (word or bigram) in input:
        score = score + parameter[feature][label]  # ADD only
    
predicted_label = argmax(scores)  # Comparison only
```

#### Learning (±1 Updates Only)
```
If prediction is wrong:
    For each feature in input:
        parameter[feature][correct_label] += 1    # INCREMENT
        parameter[feature][predicted_label] -= 1  # DECREMENT
```

#### Allowed Operations
- ✅ Addition and subtraction
- ✅ Comparisons (>, <, ==)
- ✅ Search and lookups
- ✅ Control flow (if/else, loops)

#### Not Allowed
- ❌ Matrix multiplication
- ❌ Dot products
- ❌ Linear algebra operations
- ❌ Standard neural network layers

### The 10-Word Language

| Category | Words |
|----------|-------|
| **Agents** | I, YOU |
| **Actions** | GO, GIVE, TAKE |
| **Locations** | LEFT, RIGHT, HERE |
| **Logic** | YES, NO |

These 10 words enable:
- Commands: "YOU GO LEFT"
- Statements: "I TAKE HERE"
- Responses: "YES", "NO"
- Simple dialogues: "YOU GIVE I TAKE"

### The 4 Tasks

1. **Intent Classification**: Determine the intent of a sentence
   - Labels: COMMAND, STATEMENT, CONFIRMATION, NEGATION

2. **Action Mapping**: Map sentences to concrete actions
   - Labels: MOVE_LEFT, MOVE_RIGHT, MOVE_HERE, GIVE_ITEM, TAKE_ITEM, STAY

3. **Response Generation**: Generate appropriate responses
   - Labels: YES, NO, I_GO_LEFT, I_GO_RIGHT, I_TAKE, I_GIVE, etc.

4. **Direction Detection**: Identify mentioned directions
   - Labels: LEFT, RIGHT, HERE, NONE

## 📊 Results

The model achieves high accuracy on all tasks:

- **Intent Classification**: >95% test accuracy
- **Action Mapping**: >90% test accuracy
- **Response Generation**: >85% test accuracy
- **Direction Detection**: >95% test accuracy

All with just simple counting and ±1 updates!

## 🔧 Extensibility

### Adding a New Word

```python
from extensions import LanguageExtender

extender = LanguageExtender(vocab, model, task_registry)
extender.add_new_word("UP")

# Generate training data with new word
# Retrain the model
```

### Adding a New Task

```python
extender.add_new_task(
    task_name="agent_detection",
    description="Detect which agent is mentioned",
    labels=["AGENT_I", "AGENT_YOU", "AGENT_BOTH", "AGENT_NONE"]
)

# Create training data for new task
# Train the model on new task
```

## 📈 Visualization

The notebook includes:
- Learning curves (accuracy over epochs)
- Confusion matrices for each task
- Feature importance analysis
- Per-class precision/recall/F1 scores

## 🧪 Running Tests

```bash
# Test vocabulary
python src/vocab.py

# Test tasks
python src/tasks.py

# Test model
python src/model.py

# Test training
python src/training.py

# Test evaluation
python src/evaluation.py

# Test extensions
python src/extensions.py
```

Each module has a `__main__` block with demo code.

## 📝 Example Usage

### Training a Model

```python
from vocab import get_default_vocab
from tasks import get_default_task_registry
from model import AddSubModel
from training import Trainer

# Setup
vocab = get_default_vocab()
task_registry = get_default_task_registry()
task_labels = {name: task.labels for name, task in task_registry.tasks.items()}

# Create model
model = AddSubModel(vocab_size=len(vocab), task_labels=task_labels)

# Train
trainer = Trainer(model, vocab)
history = trainer.train(train_dataset, val_dataset, num_epochs=20)
```

### Making Predictions

```python
# Convert sentence to indices
sentence = "YOU GO LEFT"
indices = vocab.sentence_to_indices(sentence)

# Predict on each task
for task_name in task_labels.keys():
    prediction = model.predict(indices, task_name)
    print(f"{task_name}: {prediction}")
```

### Evaluating Performance

```python
from evaluation import Evaluator

evaluator = Evaluator(model, vocab)
results = evaluator.evaluate_dataset(test_dataset)
evaluator.print_evaluation_report(results)
```

## 🎓 Educational Value

This project demonstrates:

1. **Machine learning fundamentals** without the complexity
2. **Interpretability** - you can inspect every parameter
3. **Learning from scratch** - no pre-trained models or libraries
4. **Algorithm design** under constraints
5. **Complete ML pipeline** from data to evaluation

Perfect for:
- Learning ML concepts
- Teaching algorithms
- Understanding what's really necessary for learning
- Building interpretable systems

## 🤝 Contributing

This is an educational project. Feel free to:
- Add more words to the vocabulary
- Define new tasks
- Implement alternative learning rules
- Improve data generation
- Add more visualizations

## 📄 License

MIT License - free to use for educational purposes.

## 🙏 Acknowledgments

Built as a demonstration that machine learning doesn't require complex mathematics - just clear thinking and simple operations.

---

**Key Insight**: The ability to learn comes from *how* we update parameters, not *what* operations we use. Addition and subtraction are enough!
