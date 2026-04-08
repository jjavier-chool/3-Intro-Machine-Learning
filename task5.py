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
from task3 import BATCH_SIZE, HIDDEN, baseline_training, train_model, evaluate

DROPOUT = [0.6, 0.3, 0.3]
DROPOUT_ENSEMBLE = [0.6, 0.3, 0.3]
ENSEMBLE_SIZE = 5

# TODO: (consider task4 of the midterm?)

def subtask1():
  baseline_training(model=FNN(hidden=HIDDEN, dropout=DROPOUT))

# Sampling for bagging, NOT disjoint
def bootstrap_indices(n) -> np.ndarray:
  return np.random.choice(n, n, replace=True)

# Ensemble Evaluation
def evaluate_ensemble(models, loader):
  correct = 0
  total = 0

  for model in models:
    model.eval()

  with torch.no_grad():
    for inputs, targets in loader:
      probs_list = []

      for model in models:
        probs = torch.sigmoid(model(inputs))
        probs_list.append(probs)

      avg_probs = torch.mean(torch.stack(probs_list), dim=0)
      outputs = avg_probs > 0.5

      total += targets.size(0)
      correct += (outputs == targets).sum().item()

  return correct / total

def subtask2():
  # Single Dropout Model
  print("\n===== DROPOUT MODEL =====")

  dataset = load_dataset()

  train_loader = DataLoader(dataset.train, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(dataset.test, batch_size=BATCH_SIZE)

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

  # Ensemble (Bagging)
  print("\n===== ENSEMBLE (BAGGING) =====")

  dataset = load_dataset()
  with perf_timer() as timer:
    models = []

    for i in range(ENSEMBLE_SIZE):
      print(f"\nTraining model {i+1}/{ENSEMBLE_SIZE}")
      indices = bootstrap_indices(len(dataset.train))
      subset = Subset(dataset.train, list(indices))
      bag_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
      model = FNN(hidden=HIDDEN, dropout=DROPOUT_ENSEMBLE)
      model = train_model(model, bag_loader)
      models.append(model)

    ensemble_acc = evaluate_ensemble(models, test_loader)

  total_time = timer.total

  print("\nENSEMBLE RESULTS:")
  print(f"Test Accuracy: {ensemble_acc:.4f}")
  print(f"Time: {total_time:.2f} sec")
  print(f"Is bagging better?: {ensemble_acc > test_acc}")

def main():
  subtask1()
  subtask2()

if __name__ == "__main__":
  main()