from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
import load_german_credit_data
import load_titanic_data

# Load dataset
X_train, y_train, X_test, y_test = load_titanic_data.load_titanic_dataset()

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
X_train, y_train, X_test, y_test = load_german_credit_data.load_german_credit_data_dataset()

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