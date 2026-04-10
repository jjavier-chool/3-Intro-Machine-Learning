#!/usr/bin/env python3

'''
Intro to Machine Learning Assignment 3
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels

Task 5: Training using Dropout Regularization.
(3 points)

The dropout regularization on a baseline FNN is to create a simplified version by randomly dropping out a subset of the edges. You may use the FNN in Task 2 and train it without regularization. It can be performed by the PyTorch function dropout. You may first read the detailed tutorial, to see how a dropout model can be created, and then complete the following two tasks:
• (1 point) Create a single dropout model and compare its performance with the baseline model. You may tune the dropout probability for each layer.
• (2 points) Create a set (≥ 5) of different dropout models, train them using bagging and compare the performance with the baseline model. You may tune the dropout probability for each layer.

Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
'''

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common import perf_timer
from task1 import load_dataset
from task2 import FNN
from task3 import BATCH_SIZE, HIDDEN, train_model, evaluate

DROPOUT = [0.5, 0.5, 0.5]

# More diverse configs (need "different models")
ENSEMBLE_HIDDEN = [
    [256, 64, 8],
    [256, 128, 8],
    [128, 64, 16],
    [512, 128, 16],
    [256, 32, 8]
]
# For now every layer gets the same dropout, but this could be configurable
ENSEMBLE_DROPOUT = [0.5, 0.6, 0.5, 0.6, 0.5]
ENSEMBLE_SIZE = 5

def subtask1(train_loader, test_loader):
  # Single Dropout Model
  print("\n===== DROPOUT MODEL =====")

  with perf_timer() as timer:
    dropout_model = FNN(hidden=HIDDEN, dropout=DROPOUT)
    dropout_model = train_model(dropout_model, train_loader)

    train_acc = evaluate(dropout_model, train_loader)
    test_acc = evaluate(dropout_model, test_loader)

  total_time = timer.total

  print("\nDROPOUT RESULTS:")
  print(f"Train Accuracy: {train_acc:.4f}")
  print(f"Test Accuracy:  {test_acc:.4f}")
  print(f"Time: {total_time:.2f} sec")

  return test_acc

# Sampling for bagging, NOT disjoint
def bootstrap_indices(n) -> np.ndarray:
  return np.random.choice(n, n, replace=True)

# Ensemble Evaluation
def evaluate_ensemble(models, loader):
  correct = 0
  total = 0

  for m in models:
    m.eval()

  with torch.no_grad():
    for inputs, targets in loader:
      avg_probs = torch.mean(
        torch.sigmoid(torch.stack([m(inputs) for m in models])), dim=0
      )
      outputs = avg_probs > 0.5

      total += targets.size(0)
      correct += (outputs == targets).sum().item()

  return correct / total

def subtask2(dataset, train_loader, test_loader):
  print("\n===== ENSEMBLE (BAGGING) =====")

  train_time = 0
  
  models = []

  ENSEMBLE = len(ENSEMBLE_HIDDEN)

  with perf_timer() as timer:
    for i, (hidden, dropout) in enumerate(zip(ENSEMBLE_HIDDEN, ENSEMBLE_DROPOUT)):
      print(f"\nTraining model {i+1}/{ENSEMBLE}")
      indices = bootstrap_indices(len(dataset.train))
      subset = Subset(dataset.train, list(indices))
      bag_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
      model = FNN(hidden=hidden, dropout=[dropout]*3)
      print(model)
      with perf_timer() as train_timer:
        model = train_model(model, bag_loader)
      train_time += train_timer.total

  ensemble_acc = evaluate_ensemble(models, test_loader)

  total_time = timer.total

  print("\nENSEMBLE RESULTS:")
  print(f"Test Accuracy: {ensemble_acc:.4f}, Test Time: {test_time:.2f} sec")
  print(f"TOTAL Time: {total_time:.2f} sec, Train Time: {train_time:.2f} sec")

  return ensemble_acc

def main(kw=""):
  dataset = load_dataset()
  train_loader = DataLoader(dataset.train, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(dataset.test, batch_size=BATCH_SIZE)

  if kw:
    try:
      # Avoid retraining the base dropout and just use a provided target accuracy
      test_acc = float(kw)
    except ValueError:
      print("Skipping base model")
      test_acc = None
  else:
    test_acc = subtask1(train_loader, test_loader)
  
  ensemble_acc = subtask2(dataset, train_loader, test_loader)
  
  if test_acc is not None:
    print(f"Is ensemble better?: {ensemble_acc > test_acc}")

# Main
if __name__ == "__main__":
  main()