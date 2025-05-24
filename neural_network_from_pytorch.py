import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import load_titanic_data
import load_german_credit_data
import numpy as np
import time

# function to count parameters and memory
def count_params_and_ram(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ram = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    print("learnable parameters:", params)
    print("vram (float32):", round(ram/1024, 2), "KB")
    return params, ram

# simple nn with 1 or 2 hidden layers and relu or sigmoid
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu"):
        super().__init__()
        layers = []
        act = nn.ReLU() if activation == "relu" else nn.Sigmoid()
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(act)
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# runs one experiment and prints metrics
def run_exp(X_train, y_train, X_test, y_test, hidden_dims=[32], activation="relu", epochs=100, lr=1.0):
    # split val set
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    # to torch
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    input_dim = X_tr.shape[1]
    output_dim = int(y_tr.max().item()) + 1
    model = SimpleMLP(input_dim, hidden_dims, output_dim, activation)
    count_params_and_ram(model)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr)
        loss = loss_fn(out, y_tr)
        loss.backward()
        optimizer.step()
        # print every 20 epochs
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            preds = out.argmax(1)
            acc = (preds == y_tr).float().mean().item()
            print(f"epoch: {epoch}, train acc: {acc:.3f}, loss: {loss.item():.3f}")
    print("training time:", round((time.time()-start_time)*1000, 2), "ms")

    model.eval()
    with torch.no_grad():
        # val
        val_logits = model(X_val)
        val_pred = val_logits.argmax(1)
        val_acc = accuracy_score(y_val.numpy(), val_pred.numpy())
        val_prec = precision_score(y_val.numpy(), val_pred.numpy())
        val_rec = recall_score(y_val.numpy(), val_pred.numpy())
        print("val acc:", round(val_acc,3), "prec:", round(val_prec,3), "rec:", round(val_rec,3))
        # test
        test_logits = model(X_test_t)
        test_pred = test_logits.argmax(1)
        test_acc = accuracy_score(y_test_t.numpy(), test_pred.numpy())
        test_prec = precision_score(y_test_t.numpy(), test_pred.numpy())
        test_rec = recall_score(y_test_t.numpy(), test_pred.numpy())
        test_loss = loss_fn(test_logits, y_test_t).item()
        print("test acc:", round(test_acc,3), "prec:", round(test_prec,3), "rec:", round(test_rec,3), "loss:", round(test_loss,3))
    print("="*35)

if __name__ == "__main__":
    # titanic dataset
    print("titanic")
    X_train, y_train, X_test, y_test = load_titanic_data.load_titanic_dataset()
    # try a few configurations
    run_exp(X_train, y_train, X_test, y_test, hidden_dims=[32], activation="relu", epochs=100)
    run_exp(X_train, y_train, X_test, y_test, hidden_dims=[64, 32], activation="sigmoid", epochs=100)

    # german credit dataset
    print("\ngerman credit")
    X_train, y_train, X_test, y_test = load_german_credit_data.load_german_credit_data_dataset()
    run_exp(X_train, y_train, X_test, y_test, hidden_dims=[32], activation="relu", epochs=100)
    run_exp(X_train, y_train, X_test, y_test, hidden_dims=[64, 32], activation="sigmoid", epochs=100)

    print("done.")
