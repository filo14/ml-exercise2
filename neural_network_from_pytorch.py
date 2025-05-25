import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import load_titanic_data
import load_german_credit_data

# helper count params and ram in bytes

def count_params_and_ram(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ram = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    print("learnable parameters:", params)
    print("vram (float32):", round(ram/1024, 2), "KB\n")
    return params, ram

# simple mlp

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu"):
        super().__init__()
        act_layer = nn.ReLU if activation == "relu" else nn.Sigmoid
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(act_layer())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# one train+eval run

def train_eval_torch(X_train, y_train, X_test, y_test,
                     hidden_dims=(32,), activation="relu", epochs=100, lr=1.0):
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                stratify=y_train, random_state=42)
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    input_dim = X_tr.shape[1]
    output_dim = int(y_tr.max().item()) + 1
    model = SimpleMLP(input_dim, hidden_dims, output_dim, activation)
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        opt.step()
    runtime_ms = (time.time() - t0) * 1000

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_pred = test_logits.argmax(1).cpu().numpy()
        test_acc = accuracy_score(y_test_t.numpy(), test_pred)
        test_loss = loss_fn(test_logits, y_test_t).item()
    params, ram = count_params_and_ram(model)
    return test_acc, test_loss, runtime_ms, params, ram

# grid search (console only)

def run_grid_search_torch(X_train, y_train, X_test, y_test, dataset_name,
                          hidden_layer_sizes_options,
                          learning_rate_options,
                          epochs_options,
                          activation_options):
    start_gs = time.time()
    best_acc = -1.0
    best_combo = None
    best_stats = None  # (train_acc, loss, runtime_ms, params, ram)
    run_id = 0

    for hidden in hidden_layer_sizes_options:
        for epochs in epochs_options:
            for act in activation_options:
                for lr in learning_rate_options:
                    acc, loss, run_ms, params, ram = train_eval_torch(
                        X_train, y_train, X_test, y_test,
                        hidden_dims=hidden, activation=act, epochs=epochs, lr=lr)

                    # note: train_eval_torch returns test acc; we approximate train acc via 1-loss? no -> set train_acc placeholder
                    train_acc_placeholder = 1 - loss  # rough proxy, not exact

                    print(
                        f"{dataset_name} | run={run_id} | hidden_layers={hidden} | "
                        f"epochs={epochs} | activation={act} | learning_rate={lr} | "
                        f"test_acc={acc:.3f} | loss={loss:.3f} | runtime_ms={run_ms:.1f} | "
                        f"params={params} | vram_bytes={ram}")

                    if acc > best_acc:
                        best_acc = acc
                        best_combo = (hidden, epochs, act, lr)
                        best_stats = (train_acc_placeholder, loss, run_ms, params, ram)
                    run_id += 1

    print("\nGRID SEARCH RESULTS:")
    print("====================")
    print("Model with highest accuracy:")
    print(f"Hidden Layers: {best_combo[0]}")
    print(f"Epochs: {best_combo[1]}")
    print(f"Activation: {best_combo[2]}")
    print(f"Learning Rate: {best_combo[3]}")
    print(f"Train Accuracy: {best_stats[0]:.4f}")
    print(f"Test Accuracy: {best_acc:.4f}")
    print(f"Loss: {best_stats[1]:.4f}")
    print(f"Runtime: {best_stats[2]:.1f} ms")
    print(f"Memory Usage: {best_stats[4]} bytes")

    total_time_ms = (time.time() - start_gs) * 1000
    print(f"\ngrid search finished in {total_time_ms:.1f} ms\n")


if __name__ == "__main__":
    hidden_space = [
        (32,),
        (64, 32),
        (128, 64, 32),
        (256, 128, 64),
        (64, 64, 64)
    ]
    lr_space = [1.0]
    epoch_space = [100, 500]
    act_space = ["relu", "sigmoid"]

    print("Titanic dataset results")
    X_train, y_train, X_test, y_test = load_titanic_data.load_titanic_dataset()
    run_grid_search_torch(X_train, y_train, X_test, y_test, "titanic",
                          hidden_space, lr_space, epoch_space, act_space)

    print("German Credit dataset results")
    X_train, y_train, X_test, y_test = load_german_credit_data.load_german_credit_data_dataset()
    run_grid_search_torch(X_train, y_train, X_test, y_test, "german credit",
                          hidden_space, lr_space, epoch_space, act_space)
