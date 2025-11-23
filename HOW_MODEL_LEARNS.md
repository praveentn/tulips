# How the Add/Sub Model Learns 🧠

A detailed explanation of the learning algorithm using **only addition and subtraction**.

---

## Table of Contents
1. [Core Concept](#core-concept)
2. [Step-by-Step Example](#step-by-step-example)
3. [Why It Works](#why-it-works)
4. [Mathematical Intuition](#mathematical-intuition)
5. [Comparison to Other Models](#comparison-to-other-models)

---

## Core Concept

The model maintains **integer score tables** that track associations between features and labels:

```
params[task][feature][label] = integer_score
```

### Key Principles

1. **Features** = patterns in the input
   - Unigrams: single words like "YOU", "GO"
   - Bigrams: word pairs like "YOU GO", "GO LEFT"

2. **Scores** = evidence for or against a label
   - Positive score = feature supports this label
   - Negative score = feature opposes this label
   - Zero = no evidence

3. **Learning** = adjusting scores based on errors
   - Correct label features: +1
   - Wrong label features: -1

---

## Step-by-Step Example

Let's trace through a complete example:

### Initial State (Before Training)

All scores start at 0:
```
params["intent_classification"][("unigram", YOU)][COMMAND] = 0
params["intent_classification"][("unigram", YOU)][STATEMENT] = 0
params["intent_classification"][("bigram", YOU, GO)][COMMAND] = 0
...
```

### Example 1: "YOU GO LEFT" → COMMAND

#### Prediction Phase

**Extract features:**
```python
features = [
    ("unigram", YOU),
    ("unigram", GO),
    ("unigram", LEFT),
    ("bigram", YOU, GO),
    ("bigram", GO, LEFT)
]
```

**Calculate scores for each label:**

For "COMMAND":
```
score = 0
score += params[("unigram", YOU)][COMMAND]      # 0
score += params[("unigram", GO)][COMMAND]       # 0
score += params[("unigram", LEFT)][COMMAND]     # 0
score += params[("bigram", YOU, GO)][COMMAND]   # 0
score += params[("bigram", GO, LEFT)][COMMAND]  # 0
# Total: 0
```

For "STATEMENT":
```
score = 0  # (same process, all zeros)
# Total: 0
```

**Prediction:** Tie! Randomly pick "STATEMENT" (wrong!)

#### Learning Phase

True label: COMMAND  
Predicted: STATEMENT

**Update rule:** ±1 for each feature

```python
# Increase scores for COMMAND (correct)
params[("unigram", YOU)][COMMAND] += 1        # 0 → 1
params[("unigram", GO)][COMMAND] += 1         # 0 → 1
params[("unigram", LEFT)][COMMAND] += 1       # 0 → 1
params[("bigram", YOU, GO)][COMMAND] += 1     # 0 → 1
params[("bigram", GO, LEFT)][COMMAND] += 1    # 0 → 1

# Decrease scores for STATEMENT (wrong)
params[("unigram", YOU)][STATEMENT] -= 1      # 0 → -1
params[("unigram", GO)][STATEMENT] -= 1       # 0 → -1
params[("unigram", LEFT)][STATEMENT] -= 1     # 0 → -1
params[("bigram", YOU, GO)][STATEMENT] -= 1   # 0 → -1
params[("bigram", GO, LEFT)][STATEMENT] -= 1  # 0 → -1
```

**After this example:**
```
COMMAND features: +1 each
STATEMENT features: -1 each
```

### Example 2: "I GO LEFT" → STATEMENT

#### Prediction Phase

**Features:**
```python
features = [
    ("unigram", I),
    ("unigram", GO),
    ("unigram", LEFT),
    ("bigram", I, GO),
    ("bigram", GO, LEFT)
]
```

**Scores:**

For "COMMAND":
```
score = 0
score += params[("unigram", I)][COMMAND]        # 0
score += params[("unigram", GO)][COMMAND]       # 1 ← from Example 1
score += params[("unigram", LEFT)][COMMAND]     # 1 ← from Example 1
score += params[("bigram", I, GO)][COMMAND]     # 0
score += params[("bigram", GO, LEFT)][COMMAND]  # 1 ← from Example 1
# Total: 3
```

For "STATEMENT":
```
score = 0
score += params[("unigram", I)][STATEMENT]        # 0
score += params[("unigram", GO)][STATEMENT]       # -1 ← from Example 1
score += params[("unigram", LEFT)][STATEMENT]     # -1 ← from Example 1
score += params[("bigram", I, GO)][STATEMENT]     # 0
score += params[("bigram", GO, LEFT)][STATEMENT]  # -1 ← from Example 1
# Total: -3
```

**Prediction:** COMMAND (wrong! should be STATEMENT)

#### Learning Phase

```python
# Increase STATEMENT scores
params[("unigram", I)][STATEMENT] += 1        # 0 → 1
params[("unigram", GO)][STATEMENT] += 1       # -1 → 0
params[("unigram", LEFT)][STATEMENT] += 1     # -1 → 0
params[("bigram", I, GO)][STATEMENT] += 1     # 0 → 1
params[("bigram", GO, LEFT)][STATEMENT] += 1  # -1 → 0

# Decrease COMMAND scores
params[("unigram", I)][COMMAND] -= 1          # 0 → -1
params[("unigram", GO)][COMMAND] -= 1         # 1 → 0
params[("unigram", LEFT)][COMMAND] -= 1       # 1 → 0
params[("bigram", I, GO)][COMMAND] -= 1       # 0 → -1
params[("bigram", GO, LEFT)][COMMAND] -= 1    # 1 → 0
```

### After Two Examples

**Pattern emerging:**
- "YOU" → positive for COMMAND
- "I" → positive for STATEMENT
- Shared words (GO, LEFT) → scores balance out

### Example 3: "YOU GO LEFT" → COMMAND (again)

**Prediction:**

For COMMAND:
```
score = 0
score += params[("unigram", YOU)][COMMAND]      # 1
score += params[("unigram", GO)][COMMAND]       # 0 (balanced)
score += params[("unigram", LEFT)][COMMAND]     # 0 (balanced)
score += params[("bigram", YOU, GO)][COMMAND]   # 1
score += params[("bigram", GO, LEFT)][COMMAND]  # 0 (balanced)
# Total: 2
```

For STATEMENT:
```
score = 0
score += params[("unigram", YOU)][STATEMENT]      # -1
score += params[("unigram", GO)][STATEMENT]       # 0
score += params[("unigram", LEFT)][STATEMENT]     # 0
score += params[("bigram", YOU, GO)][STATEMENT]   # -1
score += params[("bigram", GO, LEFT)][STATEMENT]  # 0
# Total: -2
```

**Prediction:** COMMAND ✓ (CORRECT!)

No updates needed since prediction is correct.

---

## Why It Works

### 1. **Discriminative Features Get Reinforced**

- "YOU" consistently appears in COMMAND examples
- "I" consistently appears in STATEMENT examples
- These features accumulate large positive/negative scores

### 2. **Ambiguous Features Stay Near Zero**

- "GO" appears in both COMMAND and STATEMENT
- Gets +1 for COMMAND in some examples
- Gets -1 for COMMAND in other examples
- Net effect: scores stay close to 0

### 3. **Combination of Features**

Even if individual features are weak, combinations help:
```
"YOU" alone: +1 for COMMAND
"GO" alone: 0 (ambiguous)
"YOU GO" together: +2 for COMMAND (bigram feature)
```

### 4. **Error-Driven Learning**

- Only updates when wrong
- Gradually refines decision boundaries
- Similar to perceptron convergence

---

## Mathematical Intuition

### Linear Classifier Without Multiplication

The prediction function is:
```
score(label) = Σ params[feature][label]
```

This is equivalent to a dot product, but implemented with **only addition**:
```
score = w₁ + w₂ + w₃ + ... + wₙ
```

### Perceptron-Style Updates

The update rule:
```
if prediction ≠ truth:
    params[feature][truth] += 1
    params[feature][prediction] -= 1
```

Is equivalent to:
```
w_correct += 1 * feature_vector
w_predicted -= 1 * feature_vector
```

But without the multiplication!

### Why ±1 Works

The magnitude doesn't matter for classification:
- We only care about relative scores
- ±1 is sufficient to shift preferences
- Larger updates would just converge faster (but might overshoot)

---

## Comparison to Other Models

| Model | Operations | Parameters | Interpretability |
|-------|-----------|------------|------------------|
| **Add/Sub Model** | +, - only | Integer counts | Fully interpretable |
| **Perceptron** | +, -, × | Real-valued weights | Weights hard to interpret |
| **Logistic Regression** | +, -, ×, exp, log | Real-valued weights | Weights hard to interpret |
| **Neural Network** | +, -, ×, activation | Many layers of weights | Black box |

### Advantages of Add/Sub Model

✅ **Simplicity:** Only counting and arithmetic  
✅ **Interpretability:** Can inspect every parameter  
✅ **Debuggability:** Easy to trace predictions  
✅ **Educational:** Shows core of machine learning  
✅ **No libraries needed:** Pure Python logic

### Limitations

❌ **Linearly separable only:** Can't learn XOR  
❌ **No probabilities:** Just scores, not confidences  
❌ **Fixed features:** Must predefine unigrams/bigrams  
❌ **Integer precision:** Could be limiting in some cases

---

## Summary

The model learns by:

1. **Counting** which features appear with which labels
2. **Adding** these counts to make predictions
3. **Adjusting** counts when mistakes are made

It's remarkably simple but surprisingly effective!

The key insight: **Machine learning is about tracking patterns, not complex math.**

---

## Visualization

```
Training Example: "YOU GO LEFT" → COMMAND

Before:
  Feature        COMMAND  STATEMENT
  ─────────────  ───────  ─────────
  YOU            0        0
  GO             0        0
  LEFT           0        0
  YOU_GO         0        0
  GO_LEFT        0        0
                 ───      ───
  TOTAL:         0        0
  
  Prediction: TIE (wrong if we guess STATEMENT)

After Update:
  Feature        COMMAND  STATEMENT
  ─────────────  ───────  ─────────
  YOU            1        -1        ← Learned!
  GO             1        -1
  LEFT           1        -1
  YOU_GO         1        -1        ← Learned!
  GO_LEFT        1        -1
                 ───      ───
  TOTAL:         5        -5
  
  Next time: Will predict COMMAND ✓
```

---

**The magic of machine learning isn't in complex operations—it's in how we systematically adjust simple numbers based on feedback!** 🎯
