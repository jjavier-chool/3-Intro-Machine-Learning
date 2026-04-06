'''
Note: It might be worth just incorporating all the tasks into one big file
'''
import torch.nn as nn

from tasks2_4 import baseline_training, input_dim

# Keep adjusting these
HIDDEN1 = 256
HIDDEN2 = 128
HIDDEN3 = 8

# TASK 5: FNN with dropout
class FNN(nn.Module):
  def __init__(self, input_dim):
    super(FNN, self).__init__()
    
    self.net = nn.Sequential(
      nn.Linear(input_dim, HIDDEN1),
      nn.ReLU(),
      nn.Dropout(p=0.3),
      nn.Linear(HIDDEN1, HIDDEN2),
      nn.ReLU(),
      nn.Dropout(p=0.95),
      nn.Linear(HIDDEN2, HIDDEN3),
      nn.ReLU(),
      nn.Linear(HIDDEN3, 1)
    )

  def forward(self, x):
    return self.net(x).squeeze()

if __name__ == "__main__":
  baseline_training(model = FNN(input_dim))