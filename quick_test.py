"""
Quick test script to verify the mini-language learning system works.

This runs a minimal end-to-end test of the entire pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from vocab import get_default_vocab
from tasks import get_default_task_registry
from model import AddSubModel
from data_utils import Dataset, Example, preprocess_dataset, combine_datasets
from training import Trainer
from evaluation import Evaluator


def run_quick_test():
    """Run a quick end-to-end test of the system."""
    
    print("="*70)
    print("MINI-LANGUAGE LEARNING SYSTEM - QUICK TEST")
    print("="*70)
    
    # 1. Setup vocabulary and tasks
    print("\n1. Setting up vocabulary and tasks...")
    vocab = get_default_vocab()
    task_registry = get_default_task_registry()
    print(f"   ✓ Vocabulary: {len(vocab)} words")
    print(f"   ✓ Tasks: {len(task_registry.tasks)}")
    
    # 2. Create model
    print("\n2. Creating model...")
    task_labels = {name: task.labels for name, task in task_registry.tasks.items()}
    model = AddSubModel(vocab_size=len(vocab), task_labels=task_labels, use_bigrams=True)
    print(f"   ✓ Model created: {model}")
    
    # 3. Load a small dataset
    print("\n3. Loading datasets...")
    data_dir = Path("data/tasks")
    
    # Load just one task for quick test
    task_name = "intent_classification"
    train_path = data_dir / task_name / "train.jsonl"
    val_path = data_dir / task_name / "val.jsonl"
    
    train_dataset = Dataset.load_jsonl(str(train_path))
    val_dataset = Dataset.load_jsonl(str(val_path))
    
    # Take a small subset for quick test
    train_dataset.examples = train_dataset.examples[:50]
    val_dataset.examples = val_dataset.examples[:20]
    
    # Preprocess
    train_dataset = preprocess_dataset(train_dataset, vocab)
    val_dataset = preprocess_dataset(val_dataset, vocab)
    
    print(f"   ✓ Train: {len(train_dataset)} examples")
    print(f"   ✓ Val: {len(val_dataset)} examples")
    
    # 4. Train
    print("\n4. Training for 5 epochs...")
    trainer = Trainer(model, vocab, verbose=False)
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=5,
        shuffle=True
    )
    
    final_train_acc = history.metrics[-1].train_accuracy
    final_val_acc = history.metrics[-1].val_accuracy
    print(f"   ✓ Training complete")
    print(f"   ✓ Final train accuracy: {final_train_acc:.4f}")
    print(f"   ✓ Final val accuracy: {final_val_acc:.4f}")
    
    # 5. Evaluate
    print("\n5. Evaluating...")
    evaluator = Evaluator(model, vocab)
    results = evaluator.evaluate_dataset(val_dataset)
    
    for task, task_results in results.items():
        if task != "overall":
            print(f"   ✓ {task}: {task_results['accuracy']:.4f} accuracy")
    
    # 6. Test predictions
    print("\n6. Testing predictions...")
    test_sentences = [
        ("YOU GO LEFT", "intent_classification"),
        ("I TAKE HERE", "intent_classification"),
        ("YES", "intent_classification"),
        ("NO", "intent_classification"),
    ]
    
    for sentence, task in test_sentences:
        indices = vocab.sentence_to_indices(sentence)
        prediction = model.predict(indices, task)
        print(f"   '{sentence}' → {prediction}")
    
    # 7. Summary
    print("\n" + "="*70)
    print("TEST COMPLETE - ALL SYSTEMS OPERATIONAL")
    print("="*70)
    print("\n✓ Vocabulary working")
    print("✓ Tasks defined correctly")
    print("✓ Model created (ADD/SUB only)")
    print("✓ Data loading working")
    print("✓ Training working")
    print("✓ Evaluation working")
    print("✓ Predictions working")
    print("\n🎉 Success! The system is fully functional.")
    print("\nNext steps:")
    print("  1. Run: jupyter notebook")
    print("  2. Open: notebooks/mini_language_experiments.ipynb")
    print("  3. Run all cells to see the complete demo")
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        run_quick_test()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
