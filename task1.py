'''
Intro to Machine Learning Assignment 3
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels

Task 1: Data Preparation.
(1 point)

You are required to download and transform the text reviews to term frequency-inverse document frequency (tf-idf) vectors. You may use the approach and the programs in the textbook to do so. Make sure that all text reviews are cleaned and correctly transformed. You may use 70% data as training data and the rest as test data.
'''

import joblib
import torch

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

X_full = torch.cat((X_train, X_test), dim=0)
y_full = torch.cat((y_train, y_test), dim=0)

input_dim = X_train.shape[1]