import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np
import time

"""
Intro to Machine Learning Midterm
Encompasses the solution to Tasks 2-4.
Code is adopted and modified from given tutorial in the instructions.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
# Manual tuning
HIDDEN1 = 512
HIDDEN2 = 256
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
EPOCHS = 10
BATCH_SIZE = 128
K_FOLDS = 5

# Load Data
print("Loading data...")

X_train = joblib.load('X_train.pkl')
X_test = joblib.load('X_test.pkl')
y_train = joblib.load('y_train.pkl')
y_test = joblib.load('y_test.pkl')

# Convert sparse → dense → tensor
torch.manual_seed(42)
X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

input_dim = X_train.shape[1]

# TASK 2: Define FNN
class FNN(nn.Module):
  def __init__(self, input_dim):
    super(FNN, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(input_dim, HIDDEN1),
      nn.ReLU(),
      nn.Linear(HIDDEN1, HIDDEN2),
      nn.ReLU(),
      nn.Linear(HIDDEN2, 2)
    )

  def forward(self, x):
    return self.net(x)

# TASK 3: Train Function
def train_model(model, loader):
  loss_function = nn.CrossEntropyLoss()
  optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
  )

  for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for i, data in enumerate(loader, 0):
      inputs, targets =  data

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = loss_function(outputs, targets)

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
      outputs = model(inputs)

      _, predicted = torch.max(outputs.data, 1)
      total += targets.size(0)
      correct += (predicted == targets).sum().item()
  return correct / total

# TASKS 3-4: Baseline Training
def baseline_training():
  print("\n===== BASELINE TRAINING =====")

  train_dataset = TensorDataset(X_train, y_train)
  test_dataset = TensorDataset(X_test, y_test)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

  model = FNN(input_dim)

  start_time = time.time()

  model = train_model(model, train_loader)

  train_acc = evaluate(model, train_loader)
  test_acc = evaluate(model, test_loader)

  total_time = time.time() - start_time

  print("\nBASELINE RESULTS:")
  print(f"Train Accuracy: {train_acc:.4f}")
  print(f"Test Accuracy:  {test_acc:.4f}")
  print(f"Time: {total_time:.2f} sec")

  return train_acc, test_acc, total_time

# TASK 4: K-fold Cross Validation
def kfold_training():
  print("\n===== K-FOLD CROSS VALIDATION =====")

  kf = KFold(n_splits=K_FOLDS, shuffle=True)

  train_accs = []
  val_accs = []

  start_time = time.time()

  for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")

    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    train_dataset = TensorDataset(X_tr, y_tr)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = FNN(input_dim)

    model = train_model(model, train_loader)

    train_acc = evaluate(model, train_loader)
    val_acc = evaluate(model, val_loader)

    print(f"Fold {fold+1} Train Acc: {train_acc:.4f}")
    print(f"Fold {fold+1} Val Acc:   {val_acc:.4f}")

    train_accs.append(train_acc)
    val_accs.append(val_acc)

  total_time = time.time() - start_time

  print("\nK-FOLD RESULTS:")
  print(f"Avg Train Accuracy: {np.mean(train_accs):.4f}")
  print(f"Avg Validation Accuracy: {np.mean(val_accs):.4f}")
  print(f"Time: {total_time:.2f} sec")

  return np.mean(train_accs), np.mean(val_accs), total_time

# Main
if __name__ == "__main__":
  baseline_training()
  kfold_training()
