import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
import load_german_credit_data
import load_titanic_data

def get_model_memory_usage_sklearn(model):
    """Estimates the memory usage of an sklearn MLPClassifier."""
    total_bytes = 0
    for layer in model.coefs_:
        total_bytes += layer.nbytes
    for intercept in model.intercepts_:
        total_bytes += intercept.nbytes
    return total_bytes

# Load dataset
X_train, y_train, X_test, y_test = load_titanic_data.load_titanic_dataset()

start_time = time.time()

# Create and train the MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(32), max_iter=100, random_state=42, learning_rate_init=1)
model.fit(X_train, y_train)
training_accuracy = model.score(X_train, y_train)

# Predict probabilities for test set
y_pred_prob = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Compute loss and accuracy
loss = log_loss(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)

end_time = time.time()
runtime = end_time - start_time
memory_usage = get_model_memory_usage_sklearn(model)

print("Titanic dataset results")
print(f"Runtime: {runtime:.4f} seconds")
print(f"Memory Usage (Parameters): {memory_usage:.2f}")
print(f"Training Accuracy: {training_accuracy:.4f}")
print(f"Sklearn MLP Loss: {loss:.4f}")
print(f"Sklearn MLP Accuracy: {accuracy:.4f}")
print("")

# Load dataset
X_train, y_train, X_test, y_test = load_german_credit_data.load_german_credit_data_dataset()

start_time = time.time()

# Create and train the MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(32), max_iter=100, random_state=42, learning_rate_init=1)
model.fit(X_train, y_train)
training_accuracy = model.score(X_train, y_train)

# Predict probabilities for test set
y_pred_prob = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Compute loss and accuracy
loss = log_loss(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)

end_time = time.time()
runtime = end_time - start_time
memory_usage = get_model_memory_usage_sklearn(model)

print("German Credit data dataset results")
print(f"Runtime: {runtime:.4f} seconds")
print(f"Memory Usage (Parameters): {memory_usage:.2f}")
print(f"Training Accuracy: {training_accuracy:.4f}")
print(f"Sklearn MLP Loss: {loss:.4f}")
print(f"Sklearn MLP Accuracy: {accuracy:.4f}")