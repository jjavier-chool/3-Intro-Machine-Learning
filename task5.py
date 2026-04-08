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

from task2 import FNN
from task3 import HIDDEN, baseline_training

DROPOUT = [0.6, 0.3, 0.3]

# TODO: Incorporate test2 (consider task4 of the midterm?)

def main():
  baseline_training(model=FNN(hidden=HIDDEN, dropout=DROPOUT))

if __name__ == "__main__":
  main()