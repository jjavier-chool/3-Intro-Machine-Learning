#!/usr/bin/env python3

'''
Intro to Machine Learning Assignment 3
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels

Task 3: Training a Baseline Model and Tuning the Hyperparameters.
(2 points)

You are required to tune the FNN’s structure (the previous task) and try a few values for learning rate and weight decay (L2-regularization parameter), to obtain an accuracy as high as possible or ≥ 90%. Compare the accuracy and time cost to the logistic regression model in the textbook. Please feel free to add more layers and neurons to make the FNN as powerful as possible. If the training performance is bad, please use `torch.optim.Adam` which often has a better performance than torch.optim.SGD in deep learning.
'''

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from task1 import load_dataset
from common import perf_timer

# Manual tuning
HIDDEN = [256, 64, 6]
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.0002
EPOCHS = 10
BATCH_SIZE = 128

# Train function
def train_model(model, loader, weight_decay: float=0):
  loss_function = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=weight_decay
  )

  for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for i, data in enumerate(loader, 0):
      inputs, targets =  data

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = loss_function(outputs, targets.float())

      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

  return model

# Evaluate
def evaluate(model, loader):
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for i, data in enumerate(loader, 0):
      inputs, targets = data
      outputs = torch.sigmoid(model(inputs)) > 0.5

      total += targets.size(0)
      correct += (outputs == targets).sum().item()
  return correct / total

def baseline_training(model):
  print("\n===== BASELINE TRAINING =====")

  dataset = load_dataset()

  train_loader = DataLoader(dataset.train, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(dataset.test, batch_size=BATCH_SIZE)

  print(model)
  
  with perf_timer() as timer:
    model = train_model(model, train_loader, weight_decay=WEIGHT_DECAY)

  train_acc = evaluate(model, train_loader)
  test_acc = evaluate(model, test_loader)

  total_time = timer.total

  print("\nBASELINE RESULTS:")
  print(f"Train Accuracy: {train_acc:.4f}")
  print(f"Test Accuracy:  {test_acc:.4f}")
  print(f"Time: {total_time:.2f} sec")

  return train_acc, test_acc, total_time

def main():
  from task2 import FNN
  baseline_training(FNN(hidden=HIDDEN))

if __name__ == "__main__":
  main()