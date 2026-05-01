import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load balanced dataset
df = pd.read_csv("balanced_reddit_dataset.csv")

# Use only text + label
X = df["text"]
y = df["label"]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---- Explainability Section ----
import numpy as np
import matplotlib.pyplot as plt

# Get feature names and model coefficients
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_[0]

# Top 20 depression-indicative words (label = 1)
top_positive_indices = np.argsort(coefficients)[-20:]
top_positive_words = feature_names[top_positive_indices]
top_positive_values = coefficients[top_positive_indices]

# Top 20 non-depression words (label = 0)
top_negative_indices = np.argsort(coefficients)[:20]
top_negative_words = feature_names[top_negative_indices]
top_negative_values = coefficients[top_negative_indices]

# Print ranked results
print("\nTop 20 Depression-Indicative Words (label=1):\n")
for word, coef in zip(top_positive_words[::-1], top_positive_values[::-1]):
    print(f"{word:20s} {coef:.4f}")

print("\nTop 20 Non-Depression Words (label=0):\n")
for word, coef in zip(top_negative_words, top_negative_values):
    print(f"{word:20s} {coef:.4f}")

# ---- Optional: Bar Plot Visualization ----
plt.figure(figsize=(10, 6))
plt.barh(top_positive_words, top_positive_values, color='crimson')
plt.title("Top 20 Depression-Indicative Words")
plt.xlabel("Coefficient Strength")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.barh(top_negative_words, top_negative_values, color='steelblue')
plt.title("Top 20 Non-Depression Words")
plt.xlabel("Coefficient Strength")
plt.tight_layout()
plt.show()
