import os
import tarfile
import urllib.request
import time
import sys
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import joblib

"""
Intro to Machine Learning Midterm
Encompasses the solution to Task 1.
Code is adopted and modified from the textbook.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
nltk.download('stopwords')
stop = stopwords.words('english')

DATA_URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
DATA_TAR = 'aclImdb_v1.tar.gz'
DATA_DIR = 'aclImdb'
CSV_FILE = 'movie_data.csv'

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
    max_features=20000, #Smaller
    min_df=5, #Less noise
    ngram_range=(1,2) #Better detect negation
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

# Main
if __name__ == "__main__":
  download_dataset()
  build_csv()
  process_data()
