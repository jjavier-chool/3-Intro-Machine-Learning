import time
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Data
df = pd.read_csv('movie_data.csv', encoding='utf-8')

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

# Best model from the book's GridSearchCV
tfidf = TfidfVectorizer(
    strip_accents=None,
    lowercase=False,
    preprocessor=None,
    tokenizer=str.split,
    token_pattern=None,
    ngram_range=(1,1),
    stop_words=None
)

clf = LogisticRegression(
    penalty='l2',
    C=10.0,
    solver='liblinear'
)

model = Pipeline([
    ('vect', tfidf),
    ('clf', clf)
])

# Measure Runtime
start_time = time.time()

model.fit(X_train, y_train)

train_time = time.time() - start_time
print(f"Training Time: {train_time:.2f} sec")

# Evaluate
start_time = time.time()

test_acc = model.score(X_test, y_test)

test_time = time.time() - start_time

print(f"Test Accuracy: {test_acc:.3f}")
print(f"Evaluation Time: {test_time:.2f} sec")
