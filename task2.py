'''
Intro to Machine Learning Assignment 3
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels

Task 2: Building an FNN.
(1 point)

Define an FNN classifier which takes a tf-idf vector, i.e., the vector representation of a review, and predicts the writer’s attitude: 1 for positive and 0 for negative. This task should be solved together with the next one. You need to tune your model, i.e., changing the number of layers or neurons, to see if a good accuracy can be obtained by training.
'''

import torch.nn as nn

from task1 import input_dim

# TASK 2: Define FNN
class FNN(nn.Module):
  def __init__(self, hidden: list[int], dropout: list[float]|None = None):
    super(FNN, self).__init__()
    SZ = [input_dim, *hidden]

    # Flag if dropout is even used to avoid clutter in the printout
    use_dropout = dropout is not None
    if dropout is None:
      dropout = [0.]*len(hidden)
    
    layers = []
    for i, o, p in zip(SZ, SZ[1:], dropout):
        layers.append(nn.Linear(i, o))
        layers.append(nn.ReLU())
        if use_dropout:
            layers.append(nn.Dropout(p))
    
    layers.append(nn.Linear(SZ[-1], 1))
    
    self.net = nn.Sequential(*layers) # Finally define at the end
  
  def __str__(self):
    return str(self.net)

  def forward(self, x):
    return self.net(x).squeeze()