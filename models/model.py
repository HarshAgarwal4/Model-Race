import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("models/sentiment_model.pkl")

df = pd.read_csv('data/cleaned_review.csv')

X = df['review']
Y = df['sentiment']

X_train , X_test , Y_train , Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))
print("Vocabulary :", vectorizer.get_feature_names_out() )

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    C=5
)

start_time = time.perf_counter()

model.fit(X_train_tfidf, Y_train)

training_time = time.perf_counter() - start_time

print("\nModel training completed!")
print(f"Training time: {training_time:.4f} seconds")

Y_pred = model.predict(X_test_tfidf)

print(Y_pred)
print(len(Y_pred))

accuracy = accuracy_score(Y_test, Y_pred)

precision = precision_score(
    Y_test,
    Y_pred,
    zero_division=0
)

recall = recall_score(
    Y_test,
    Y_pred,
    zero_division=0
)

f1 = f1_score(
    Y_test,
    Y_pred,
    zero_division=0
)

print("\n========== MODEL EVALUATION ==========")

print(f"AccuracY:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n--- Classification Report ---")

print(
    classification_report(
        Y_test,
        Y_pred,
        target_names=["Negative", "Positive"],
        zero_division=0
    )
)

print("\n--- Confusion Matrix ---")

print(
    confusion_matrix(
        Y_test,
        Y_pred,
        labels=[0, 1]
    )
)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "vectorizer": vectorizer,
        "model": model,
        "labels": {
            0: "negative",
            1: "positive"
        }
    },
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Model location:", MODEL_PATH)