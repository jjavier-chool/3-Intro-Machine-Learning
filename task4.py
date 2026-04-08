#!/usr/bin/env python3

'''
Intro to Machine Learning Assignment 3
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels

Task 4: Training using k-Fold Cross Validation.
(3 points)

We learned how to use k-fold cross validation in scikit-learn. But the PyTorch training functions does not directly support k-fold cross validation. In order to use this technique, you are required to implement a basic k-fold cross validation in training an FNN. You may first read the tutorial and then adapt the code for your program. Comparing the performance (time cost, training accuracy and test accuracy) of the training results with and without k-fold cross validation. You may tune the value of k for a good performance.

Reference: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
'''

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np

from common import perf_timer
from task1 import X_full, y_full
from task2 import FNN
from task3 import HIDDEN, BATCH_SIZE, train_model, evaluate

K_FOLDS = 5

# TASK 4: K-fold Cross Validation
def kfold_training():
  print("\n===== K-FOLD CROSS VALIDATION =====")

  kf = KFold(n_splits=K_FOLDS, shuffle=True)

  train_accs = []
  val_accs = []

  with perf_timer() as timer:
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
        print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")

        X_tr = X_full[train_idx]
        y_tr = y_full[train_idx]
        X_val = X_full[val_idx]
        y_val = y_full[val_idx]

        train_dataset = TensorDataset(X_tr, y_tr)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        model = FNN(hidden=HIDDEN)

        model = train_model(model, train_loader)

        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)

        print(f"Fold {fold+1} Train Acc: {train_acc:.4f}")
        print(f"Fold {fold+1} Val Acc:   {val_acc:.4f}")

        train_accs.append(train_acc)
        val_accs.append(val_acc)

  total_time = timer.total

  print("\nK-FOLD RESULTS:")
  print(f"Avg Train Accuracy: {np.mean(train_accs):.4f}")
  print(f"Avg Validation Accuracy: {np.mean(val_accs):.4f}")
  print(f"Time: {total_time:.2f} sec")

  return np.mean(train_accs), np.mean(val_accs), total_time

def main():
    kfold_training()

if __name__ == "__main__":
    main()