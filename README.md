# SSD Performance Predictor usign PyTorch

A neural network that predicts SSD (Solid State Drive) performance metrics from telemetry measurements, useful for early warning systems, capacity planning, and storage health monitoring.

## The Problem

An SSD constantly emits health signals:

| Feature | Description |
|---------|-------------|
| `queue_depth` | How many tasks are waiting |
| `temperature` | Drive operating temperature |
| `latency` | Response time |
| `read_ratio` | Read vs write ratio |
| `cache_hits` | Cache efficiency |
| `wear_level` | Drive wear percentage |

Given these 6 measurements, predict a performance metric like **throughput**.

This is a **regression** problem — predicting a continuous number, not a category.

## Architecture

```
Input (6 features)
      ↓
Linear(6 → 64) + ReLU      ← learn patterns in raw measurements
      ↓
Dropout(0.3)               ← randomly zero 30% of neurons (prevents overfitting)
      ↓
Linear(64 → 64) + ReLU     ← refine patterns
      ↓
Linear(64 → 1)             ← single prediction (no ReLU — output can be negative)
      ↓
Predicted performance metric
```

**Total parameters:** 4,609

## Key Concepts

**Why MSELoss?**
MSE = Mean Squared Error = average of (prediction − target)². Used for regression. For classification, use CrossEntropyLoss instead.

**Why Dropout?**
Randomly zeros 30% of neurons during training — forces the network to not rely on any single neuron. Automatically disabled during `model.eval()`.

**Why save based on validation loss?**
Training loss always decreases (the model memorizes training data). Validation loss tells us if the model generalizes to new, unseen data. We save the checkpoint with the lowest validation loss.

**Train / Val / Test split:**
| Split | Size | Purpose |
|-------|------|---------|
| Train | 80% | Model learns from this |
| Val | 10% | Pick best model during training |
| Test | 10% | Final honest evaluation — touch only once |

## Project Structure

```
ssd-predictor/
├── ssd_predictor.py   # Full implementation + training script
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python ssd_predictor.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+

## Author

Niloofar Tavahoodi — M.A.Sc. Candidate, Electrical & Computer Engineering, University of Victoria
