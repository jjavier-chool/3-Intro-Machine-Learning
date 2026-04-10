import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

# Hyperparameters
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 1e-7
EPOCHS = 10
BATCH_SIZE = 128
ENSEMBLE_SIZE = 5

# More diverse configs (need "different models")
HIDDEN_CONFIGS = [
    [256, 64, 8],
    [256, 128, 64],
    [128, 64, 16],
    [512, 128, 16],
    [256, 32, 8]
]

DROPOUT_RATES = [0.5, 0.5, 0.5, 0.5, 0.5]

# Load Data
print("Loading data...")

X_train = joblib.load('X_train.pkl')
X_test = joblib.load('X_test.pkl')
y_train = joblib.load('y_train.pkl')
y_test = joblib.load('y_test.pkl')

torch.manual_seed(42)

X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

input_dim = X_train.shape[1]

# Bagging
def bootstrap_indices(n):
  return np.random.choice(n, size=n, replace=True)

# Dropout Model
class DropoutFNN(nn.Module):
  def __init__(self, input_dim, hidden_layers, dropout_p):
    super(DropoutFNN, self).__init__()

    layers = []
    SZ = [input_dim] + hidden_layers

    for i, o in zip(SZ, SZ[1:]):
      layers.append(nn.Linear(i, o))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(p=dropout_p))

    layers.append(nn.Linear(SZ[-1], 1))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x).squeeze()

# Train
def train_model(model, loader):
  loss_function = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
  )

  for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for i, data in enumerate(loader, 0):
      inputs, targets = data

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = loss_function(outputs, targets.float())

      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

  return model

# Single model eval
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

# Averaging votes
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

# Main
if __name__ == "__main__":
  # Single Dropout Model
  print("\n===== DROPOUT MODEL =====")

  dropout_model = DropoutFNN(
    input_dim,
    HIDDEN_CONFIGS[0], # use first config
    DROPOUT_RATES[0]
  )

  train_dataset = TensorDataset(X_train, y_train)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

  start_time = time.time()
  dropout_model = train_model(dropout_model, train_loader)
  train_time = time.time() - start_time

  train_acc = evaluate(dropout_model, train_loader)
  start_time = time.time()
  test_acc = evaluate(dropout_model, test_loader)
  test_time = time.time() - start_time

  print("\nDROPOUT RESULTS:")
  print(f"Train Accuracy: {train_acc:.4f}, Train Time: {train_time:.2f} sec")
  print(f"Test Accuracy:  {test_acc:.4f}, Test Time: {test_time:.2f} sec")

  # Ensemble (Bagging)
  print("\n===== ENSEMBLE (BAGGING) =====")

  models = []
  train_time = 0

  start_time = time.time()
  for i in range(ENSEMBLE_SIZE):
    print(f"\nTraining model {i+1}/{ENSEMBLE_SIZE}")

    #torch.manual_seed(i)
    #np.random.seed(i)

    idx = bootstrap_indices(len(X_train))

    subset = TensorDataset(X_train[idx], y_train[idx])
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

    model = DropoutFNN(
      input_dim,
      HIDDEN_CONFIGS[i],
      DROPOUT_RATES[i]
    )

    now = time.time()
    model = train_model(model, loader)
    train_time += time.time() - now
    models.append(model)
    train_acc = evaluate(model, train_loader)

    print(f"Model {i+1} Train Acc: {train_acc:.4f}")

  now = time.time()
  ensemble_acc = evaluate_ensemble(models, test_loader)
  test_time = time.time() - now

  total_time = time.time() - start_time

  print("\nENSEMBLE RESULTS:")
  print(f"Test Accuracy: {ensemble_acc:.4f}, Test Time: {test_time:.2f} sec")
  print(f"TOTAL Time: {total_time:.2f} sec, Train Time: {train_time:.2f} sec")
  print(f"Is ensemble better?: {ensemble_acc > test_acc}")
