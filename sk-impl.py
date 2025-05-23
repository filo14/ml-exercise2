import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
import pandas as pd

def load_titanic_dataset():
    # Load the scaled training features
    X_train_df = pd.read_csv('./titanic/titanic_X_train_scaled.csv')
    X_train = X_train_df.to_numpy().astype('float32')

    # If you have the labels in a separate CSV (e.g., titanic_y_train.csv)
    y_train_df = pd.read_csv('./titanic/titanic_y_train.csv')
    y_train = y_train_df['Survived'].to_numpy().astype('int32').reshape(-1)

    # Similarly for the test set
    X_test_df = pd.read_csv('./titanic/titanic_X_test_scaled.csv')
    X_test = X_test_df.to_numpy().astype('float32')

    y_test_df = pd.read_csv('./titanic/titanic_y_test.csv')
    y_test = y_test_df['Survived'].to_numpy().astype('int32').reshape(-1)

    return X_train, y_train, X_test, y_test


def load_german_credit_data_dataset():
    X_train_df = pd.read_csv('./german_credit_data/german_X_train_scaled.csv')
    X_train = X_train_df.to_numpy().astype('float32')

    y_train_df = pd.read_csv('./german_credit_data/german_y_train.csv')
    y_train = y_train_df['credit_rating'].to_numpy().astype('int32').reshape(-1)

    X_test_df = pd.read_csv('./german_credit_data/german_X_test_scaled.csv')
    X_test = X_test_df.to_numpy().astype('float32')

    y_test_df = pd.read_csv('./german_credit_data/german_y_test.csv')
    y_test = y_test_df['credit_rating'].to_numpy().astype('int32').reshape(-1)

    return X_train, y_train, X_test, y_test


# Load dataset
X_train, y_train, X_test, y_test = load_titanic_dataset()

# Create and train the MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(32), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict probabilities for test set
y_pred_prob = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Compute loss and accuracy
loss = log_loss(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)
print("Titanic dataset results")
print(f"Sklearn MLP Loss: {loss:.4f}")
print(f"Sklearn MLP Accuracy: {accuracy:.4f}")
print("\n-------------------------------\n")

# Load dataset
X_train, y_train, X_test, y_test = load_german_credit_data_dataset()

# Create and train the MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(32), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict probabilities for test set
y_pred_prob = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Compute loss and accuracy
loss = log_loss(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)

print("German Credit data dataset results")
print(f"Sklearn MLP Loss: {loss:.4f}")
print(f"Sklearn MLP Accuracy: {accuracy:.4f}")