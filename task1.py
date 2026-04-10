#!/usr/bin/env python3

'''
Intro to Machine Learning Assignment 3
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels

Task 1: Data Preparation.
(1 point)

You are required to download and transform the text reviews to term frequency-inverse document frequency (tf-idf) vectors. You may use the approach and the programs in the textbook to do so. Make sure that all text reviews are cleaned and correctly transformed. You may use 70% data as training data and the rest as test data.
'''

import os
import tarfile
import urllib.request
import time
import sys
import warnings
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import joblib
import torch
import numpy as np
from torch.utils.data import ConcatDataset, TensorDataset

_initialized = False

def init():
  global _initialized
  if _initialized: return False
  _initialized = False

  warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")
  warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
  torch.sparse.check_sparse_tensor_invariants(True)

  # TODO: Find a better place for this
  torch.manual_seed(42)

  return True

DATA_URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
DATA_TAR = 'aclImdb_v1.tar.gz'
DATA_DIR = 'aclImdb'
CSV_FILE = 'movie_data.csv'

input_dim = 20000 #Smaller?

start_time: float

# Download Dataset
def download_dataset():
  if os.path.exists(DATA_DIR):
    print("Dataset already extracted.")
    return

  if not os.path.exists(DATA_TAR):
    print("Downloading dataset...")

    def reporthook(count, block_size, total_size):
      global start_time
      if count == 0:
        start_time = time.time()
        return
      duration = time.time() - start_time
      progress_size = int(count * block_size)
      speed = progress_size / (1024.**2 * duration)
      percent = count * block_size * 100. / total_size

      sys.stdout.write(
        f'\r{int(percent)}% | {progress_size / (1024.**2):.2f} MB '
        f'| {speed:.2f} MB/s | {duration:.2f} sec'
      )
      sys.stdout.flush()

    urllib.request.urlretrieve(DATA_URL, DATA_TAR, reporthook)
    print("\nDownload complete.")

  print("Extracting dataset...")
  with tarfile.open(DATA_TAR, 'r:gz') as tar:
    tar.extractall() #Keep in mind that this takes a long while, it's 100019 extractions.
  print("Extraction complete.")

# Build CSV
def build_csv():
  if os.path.exists(CSV_FILE):
    print("CSV already exists.")
    return

  print("Building CSV file...")

  labels = {'pos': 1, 'neg': 0}
  df = pd.DataFrame()

  for s in ('test', 'train'):
    for l in ('pos', 'neg'):
      path = os.path.join(DATA_DIR, s, l)
      for file in os.listdir(path):
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
          text = f.read()
        df = pd.concat(
          [df, pd.DataFrame([[text, labels[l]]],
          columns=['review', 'sentiment'])],
          ignore_index=True
        )

  #Shuffle
  df = df.sample(frac=1, random_state=42).reset_index(drop=True)

  df.to_csv(CSV_FILE, index=False, encoding='utf-8')
  print("df:", df.shape)
  print("CSV created.")

# Tokenizer
def tokenizer(text):
  stop = stopwords.words('english')
  text = re.sub('<[^>]*>', ' ', text)
  emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
  text = re.sub(r'[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
  return [w for w in text.split() if w not in stop]

# TF-IDF Processing
def process_data():
  print("Loading CSV...")
  df = pd.read_csv(CSV_FILE, encoding='utf-8')
  print("df:", df.shape)

  X = df['review'].values
  y = df['sentiment'].values

  print("Splitting data (70/30)...")
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
  )

  print("Vectorizing text with TF-IDF...")
  tfidf = TfidfVectorizer(
    tokenizer=tokenizer,
    token_pattern=None, #To ignore warning message about being unused
    lowercase=False,
    max_features=input_dim,
    min_df=5, #Less noise?
    ngram_range=(1,2) #Better detect negation???
  )

  X_train_tfidf = tfidf.fit_transform(X_train)
  X_test_tfidf = tfidf.transform(X_test)

  print("Shapes (for sanity checking):")
  print("X_train:", X_train_tfidf.shape)
  print("X_test:", X_test_tfidf.shape)

  #Save outputs
  print("Saving processed data...")
  joblib.dump(X_train_tfidf, 'X_train.pkl')
  joblib.dump(X_test_tfidf, 'X_test.pkl')
  joblib.dump(y_train, 'y_train.pkl')
  joblib.dump(y_test, 'y_test.pkl')

  print("Task 1 completed successfully.")

# Densifying sparse matrix quickly causes OOM issues
# CSR: Compressed Sparse Row
# COO: COOrdinate Matrix
# Kept just in case
def csr_to_torch_coo(X):
  coo = X.tocoo()
  indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
  values = torch.from_numpy(coo.data).float()
  
  return torch.sparse_coo_tensor(
      indices=indices,
      values=values,
      size=coo.shape
  )

def csr_to_torch_csr(X):
  '''Convert scipy CSR to torch CSR'''
  row_indices = torch.from_numpy(X.indptr).long()
  col_indices = torch.from_numpy(X.indices).long()
  values = torch.from_numpy(X.data).float()
  return torch.sparse_csr_tensor(
      crow_indices=row_indices,
      col_indices=col_indices,
      values=values,
      size=X.shape
  )

class AclIMDB:
  '''Container for data loaded from the acl IMDB dataset.'''

  train: TensorDataset
  test: TensorDataset
  full: ConcatDataset[TensorDataset]

  def __init__(self, X_train, y_train, X_test, y_test):
    X_train = csr_to_torch_csr(X_train).to_dense()
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = csr_to_torch_csr(X_test).to_dense()
    y_test = torch.tensor(y_test, dtype=torch.long)

    self.train = TensorDataset(X_train, y_train)
    self.test = TensorDataset(X_test, y_test)
    self.full = ConcatDataset([self.train, self.test])

# Cache these so subsequent tasks don't have to reload them
X_train = None
y_train = None
X_test = None
y_test = None

def load_dataset():
  '''Load the acl IMDB dataset, integral to task1.'''

  global X_train, y_train, X_test, y_test

  if init():
    print("Loading data...")

    X_train = joblib.load('X_train.pkl')
    y_train = joblib.load('y_train.pkl')
    X_test = joblib.load('X_test.pkl')
    y_test = joblib.load('y_test.pkl')

  return AclIMDB(X_train, y_train, X_test, y_test)

# When called, force download
def main():
  download_dataset()
  build_csv()
  nltk.download('stopwords')
  process_data()

if __name__ == "__main__":
  main()