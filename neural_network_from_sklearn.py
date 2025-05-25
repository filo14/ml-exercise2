import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
import load_german_credit_data
import load_titanic_data


def get_model_memory_usage_sklearn(model):
    """returns memory used by weights and biases"""
    total_bytes = 0
    for layer in model.coefs_:
        total_bytes += layer.nbytes
    for intercept in model.intercepts_:
        total_bytes += intercept.nbytes
    return total_bytes

# grid search

def run_grid_search_sklearn(
        X_train,
        y_train,
        X_test,
        y_test,
        dataset_name,
        hidden_layer_sizes_options,
        learning_rate_options,
        max_iter_options):
    

    start_time_grid_search = time.time()

    best_acc = -1.0
    best_params = None
    best_stats = None  # (train_acc, loss, runtime_ms, mem)
    run_id = 0

    for hidden in hidden_layer_sizes_options:
        for epochs in max_iter_options:
            for lr in learning_rate_options:

                t0 = time.time()
                model = MLPClassifier(hidden_layer_sizes=hidden,
                                      max_iter=epochs,
                                      random_state=42,
                                      learning_rate_init=lr)
                model.fit(X_train, y_train)
                train_acc = model.score(X_train, y_train)

                y_prob = model.predict_proba(X_test)
                y_pred = model.predict(X_test)

                loss = log_loss(y_test, y_prob)
                acc = accuracy_score(y_test, y_pred)
                t1 = time.time()

                runtime_ms = (t1 - t0) * 1000
                mem = get_model_memory_usage_sklearn(model)

                # print line
                print(
                    f"{dataset_name} | run={run_id} | hidden_layers={hidden} | "
                    f"epochs={epochs} | learning_rate={lr} | "
                    f"train_acc={train_acc:.3f} | test_acc={acc:.3f} | "
                    f"loss={loss:.3f} | runtime_ms={runtime_ms:.1f} | mem_bytes={mem}"
                )

                if acc > best_acc:
                    best_acc = acc
                    best_params = (hidden, epochs, lr)
                    best_stats = (train_acc, loss, runtime_ms, mem)

                run_id += 1

    print("\nGRID SEARCH RESULTS:")
    print("====================")
    print("Model with highest accuracy:")
    print(f"Hidden Layers: {best_params[0]}")
    print(f"Epochs: {best_params[1]}")
    print(f"Learning Rate: {best_params[2]}")
    print(f"Train Accuracy: {best_stats[0]:.4f}")
    print(f"Test Accuracy: {best_acc:.4f}")
    print(f"Loss: {best_stats[1]:.4f}")
    print(f"Runtime: {best_stats[2]:.1f}")
    print(f"Memory Usage: {best_stats[3]}")
    total_time_ms = (time.time() - start_time_grid_search) * 1000

    print(f"\ngrid search finished in {total_time_ms:.1f} ms")



# search space

grid_hidden_layer_sizes = [
    (32,),
    (64, 32),
    (128, 64, 32),
    (256, 128, 64),
    (64, 64, 64)
]

grid_learning_rates = [1]

grid_max_iters = [100, 500, 1000]

# titanic baseline

X_train, y_train, X_test, y_test = load_titanic_data.load_titanic_dataset()

start_time = time.time()
model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=100, random_state=42, learning_rate_init=1)
model.fit(X_train, y_train)
train_acc = model.score(X_train, y_train)

y_prob = model.predict_proba(X_test)
y_pred = model.predict(X_test)

loss = log_loss(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)

end_time = time.time()
runtime = end_time - start_time
memory_usage = get_model_memory_usage_sklearn(model)

print("\nTitanic dataset results")
print(f"Runtime={runtime:.4f}")
print(f"Memory Usage={memory_usage:.2f}")
print(f"Training Accuracy={train_acc:.4f}")
print(f"Loss={loss:.4f}")
print(f"Test Accuracy={accuracy:.4f}\n")

run_grid_search_sklearn(X_train, y_train, X_test, y_test, "titanic",
                        grid_hidden_layer_sizes, grid_learning_rates, grid_max_iters)

# german credit baseline

X_train, y_train, X_test, y_test = load_german_credit_data.load_german_credit_data_dataset()

start_time = time.time()
model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=100, random_state=42, learning_rate_init=1)
model.fit(X_train, y_train)
train_acc = model.score(X_train, y_train)

y_prob = model.predict_proba(X_test)
y_pred = model.predict(X_test)

loss = log_loss(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)

end_time = time.time()
runtime = end_time - start_time
memory_usage = get_model_memory_usage_sklearn(model)

print("\nGerman credit dataset results")
print(f"Runtime={runtime:.4f}")
print(f"Memory Usage={memory_usage:.2f}")
print(f"Training Accuracy={train_acc:.4f}")
print(f"Loss={loss:.4f}")
print(f"Test Accuracy={accuracy:.4f}\n")

run_grid_search_sklearn(X_train, y_train, X_test, y_test, "german credit",
                        grid_hidden_layer_sizes, grid_learning_rates, grid_max_iters)
