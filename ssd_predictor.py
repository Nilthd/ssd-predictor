"""
SSD Performance Predictor — PyTorch
=====================================
A neural network that predicts SSD (Solid State Drive) performance
metrics from telemetry measurements.

WHAT THIS CODE DOES (explain like I'm 5):
------------------------------------------
Imagine an SSD (your computer's storage drive) is constantly sending
health signals:
    - How hot is it running?
    - How full is the queue of waiting tasks?
    - How fast is it responding?
    - How worn out is it?

This model reads those 6 signals and predicts a performance metric
(like throughput) — useful for early warning systems, capacity
planning, and storage health monitoring.

This is a REGRESSION problem — we predict a continuous number,
not a category. That's why we use MSELoss instead of CrossEntropyLoss.

Architecture:
    Input (6) → Linear → ReLU → Dropout → Linear → ReLU → Linear → Output (1)

Author: Niloofar Tavahoodi
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ── PART 1: Dataset ───────────────────────────────────────────────────────────
#
# PyTorch's DataLoader needs a Dataset object to know:
#   - how many samples exist   (__len__)
#   - how to get one sample    (__getitem__)
#
# Think of Dataset as a filing cabinet.
# DataLoader is the person pulling files out in batches of 32.

class SSDDataset(Dataset):
    """
    Wraps SSD telemetry features and labels into a PyTorch Dataset.

    Args:
        features : numpy array, shape (num_samples, 6)
                   columns: queue_depth, temperature, latency,
                            read_ratio, cache_hits, wear_level
        labels   : numpy array, shape (num_samples, 1)
                   the performance metric to predict (e.g. throughput)
    """

    def __init__(self, features, labels):
        # convert numpy arrays → PyTorch tensors
        # FloatTensor = 32-bit float — what neural networks expect
        self.features = torch.FloatTensor(features)
        self.labels   = torch.FloatTensor(labels)

    def __len__(self):
        # PyTorch calls this to know how many samples exist
        return len(self.features)

    def __getitem__(self, idx):
        # PyTorch calls this to get ONE sample at index idx
        # returns a (features, label) pair
        return self.features[idx], self.labels[idx]


# ── PART 2: Model ─────────────────────────────────────────────────────────────
#
# A 3-layer feedforward neural network (MLP — Multi-Layer Perceptron).
#
# Why 3 layers?
#   - Layer 1: learns low-level patterns in the 6 raw measurements
#   - Layer 2: combines those patterns into higher-level features
#   - Layer 3: maps features → final single prediction
#
# Data shape through the network (batch_size=32):
#   Input          : (32, 6)
#   After layer1   : (32, 64)
#   After dropout  : (32, 64)  ← 30% neurons zeroed out
#   After layer2   : (32, 64)
#   After layer3   : (32, 1)   ← one prediction per sample

class SSDModel(nn.Module):
    """
    3-layer MLP for SSD performance regression.

    Args:
        input_size  : number of input features (default 6)
        hidden_size : neurons in hidden layers (default 64)
        output_size : prediction size (default 1 for regression)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        # ALWAYS required — runs PyTorch's internal setup

        # Layer 1: raw measurements → hidden representation
        # W shape: (hidden_size, input_size) = (64, 6) = 384 weights
        self.layer1 = nn.Linear(input_size, hidden_size)

        # Layer 2: refine patterns from layer 1
        # W shape: (hidden_size, hidden_size) = (64, 64) = 4096 weights
        self.layer2 = nn.Linear(hidden_size, hidden_size)

        # Layer 3: combine everything into one prediction
        # W shape: (output_size, hidden_size) = (1, 64) = 64 weights
        self.layer3 = nn.Linear(hidden_size, output_size)

        # ReLU activation: f(x) = max(0, x)
        # Keeps positive values, zeros out negatives.
        # Without this, all linear layers collapse into one — pointless.
        self.relu = nn.ReLU()

        # Dropout: randomly zeros 30% of neurons during training.
        # Forces the network to NOT rely on any single neuron.
        # Automatically DISABLED when model.eval() is called.
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines how data flows through the network.
        PyTorch calls this automatically when you do model(input).

        Args:
            x : input tensor (batch_size, input_size)

        Returns:
            predictions : (batch_size, output_size)
        """
        x = self.relu(self.layer1(x))   # 6 → 64, apply ReLU
        x = self.dropout(x)             # randomly zero 30% of neurons
        x = self.relu(self.layer2(x))   # 64 → 64, apply ReLU
        x = self.layer3(x)              # 64 → 1, NO ReLU (output can be negative)
        return x


# ── PART 3: Training Function ─────────────────────────────────────────────────
#
# One full pass through the training data = one "epoch".
#
# For each batch of 32 samples:
#   1. zero_grad   → clear old gradients (ALWAYS first)
#   2. forward     → predict
#   3. loss        → measure how wrong we are
#   4. backward    → compute gradients (which weights caused the error?)
#   5. step        → nudge all weights to reduce the loss
#
# Returns the average loss across all batches.

def train(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """One full training pass. Returns average loss."""

    model.train()       # activates dropout
    total_loss = 0.0

    for features, labels in loader:
        optimizer.zero_grad()           # clear gradients from previous batch

        output = model(features)        # forward pass: (32,6) → (32,1)
        loss   = criterion(output, labels)  # MSE: mean((pred - true)²)

        loss.backward()                 # backprop: compute all gradients
        optimizer.step()                # update all weights

        total_loss += loss.item()       # .item() converts tensor → Python float

    return total_loss / len(loader)     # average loss across batches


# ── PART 4: Validation Function ───────────────────────────────────────────────
#
# Same as training BUT:
#   - model.eval()      → dropout OFF, batch norm fixed
#   - torch.no_grad()   → don't compute gradients (saves memory)
#   - NO backward/step  → weights never updated here
#
# WHY VALIDATE SEPARATELY?
# Training loss always goes down (model memorizes training data).
# Validation loss tells us if the model GENERALIZES to new data.
# If train loss low but val loss high → overfitting.

def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
) -> float:
    """One full validation pass. Returns average loss. No weight updates."""

    model.eval()        # disables dropout
    total_loss = 0.0

    with torch.no_grad():   # no gradients needed → faster + less memory
        for features, labels in loader:
            output = model(features)
            loss   = criterion(output, labels)
            total_loss += loss.item()

    return total_loss / len(loader)


# ── PART 5: Main ──────────────────────────────────────────────────────────────

def main():

    # ── Data ──────────────────────────────────────────────────────────────────
    # Synthetic SSD telemetry — replace with real data in production
    # Each sample has 6 features:
    # [queue_depth, temperature, latency, read_ratio, cache_hits, wear_level]
    np.random.seed(42)
    X = np.random.randn(1000, 6).astype(np.float32)
    y = np.random.randn(1000, 1).astype(np.float32)

    # Split chronologically: 80% train / 10% val / 10% test
    X_train, y_train = X[:800],    y[:800]
    X_val,   y_val   = X[800:900], y[800:900]
    X_test,  y_test  = X[900:],    y[900:]

    # Wrap in Dataset and DataLoader
    train_loader = DataLoader(SSDDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(SSDDataset(X_val,   y_val),   batch_size=32, shuffle=False)
    test_loader  = DataLoader(SSDDataset(X_test,  y_test),  batch_size=32, shuffle=False)

    # ── Model Setup ───────────────────────────────────────────────────────────
    model = SSDModel(input_size=6, hidden_size=64, output_size=1)

    criterion = nn.MSELoss()
    # MSELoss = Mean Squared Error = average of (prediction - target)²
    # use for regression (predicting a number)
    # use CrossEntropyLoss for classification (predicting a category)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,           # learning rate — how big each weight update is
        weight_decay=1e-4,  # L2 regularization — penalizes large weights
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,   # every 10 epochs...
        gamma=0.5,      # halve the learning rate
        # epoch 0-9:   lr = 0.001
        # epoch 10-19: lr = 0.0005
        # epoch 20-29: lr = 0.00025
        # big steps early (learn fast) → small steps later (fine-tune)
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    print("=" * 55)
    print("  SSD Performance Predictor — Training")
    print("=" * 55)
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    best_val_loss = float('inf')  # start at infinity — any loss will be smaller

    for epoch in range(100):

        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss   = validate(model, val_loader, criterion)
        scheduler.step()    # decay learning rate once per epoch

        # save model whenever validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            # state_dict() = all weight tensors as a dictionary
            # save based on VAL loss, not train loss:
            # train loss always decreases (memorization)
            # val loss reflects true generalization

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    # Load the best weights and evaluate on the held-out test set.
    # This number = our honest estimate of real-world performance.
    # Touch the test set ONLY ONCE — at the very end.
    model.load_state_dict(torch.load('best_model.pt'))
    test_loss = validate(model, test_loader, criterion)

    print(f"\n  Best Val Loss  : {best_val_loss:.4f}")
    print(f"  Test Loss      : {test_loss:.4f}")
    print(f"  Model saved    : best_model.pt")


if __name__ == "__main__":
    main()
