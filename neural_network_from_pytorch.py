# pytorch nn for titanic + german credit datasets
# code and comments are all lower case, just like a typical uni student
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import load_titanic_data
import load_german_credit_data
import time


# define a mlp
class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out, act="relu"):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        if act == "relu":
            self.act1 = nn.ReLU()
        elif act == "tanh":
            self.act1 = nn.Tanh()
        elif act == "sigmoid":
            self.act1 = nn.Sigmoid()
        else:
            self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(n_hid, n_out)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


# count params and estimate ram (float32)
def param_count_and_ram(model):
    p = sum([w.numel() for w in model.parameters() if w.requires_grad])
    ram = sum([w.numel() * w.element_size() for w in model.parameters() if w.requires_grad])
    print(f"total parameters: {p}")
    print(f"estimated vram usage (float32): {ram/1024:.2f} KB ({ram/1024/1024:.2f} MB)")
    return p, ram

# function to train and evaluate model (prints similar info as nn from scratch code)
def train_and_report(X_tr, y_tr, X_val, y_val, X_test, y_test, n_hid=32, act="relu", epochs=100):
    n_in = X_tr.shape[1]
    n_out = int(y_tr.max() + 1)
    model = MLP(n_in, n_hid, n_out, act)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    print(f"\nmlp config: hidden={n_hid}, activation={act}, epochs={epochs}")
    pcount, vram = param_count_and_ram(model)
    # timing
    start_ts = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tr)
        loss = loss_fn(outputs, y_tr)
        loss.backward()
        optimizer.step()
        if epoch % (epochs//10) == 0 or epoch == epochs - 1:
            _, preds = torch.max(outputs, 1)
            acc = (preds == y_tr).float().mean().item()
            print(f"epoch {epoch:04d}: train loss={loss.item():.4f} train acc={acc:.4f}")
    train_time_ms = (time.perf_counter() - start_ts) * 1000
    print(f"total training time: {train_time_ms:.2f} ms")
    # eval val
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_pred = val_logits.argmax(dim=1)
        val_acc = accuracy_score(y_val.cpu(), val_pred.cpu())
        val_prec = precision_score(y_val.cpu(), val_pred.cpu())
        val_rec = recall_score(y_val.cpu(), val_pred.cpu())
        print(f"validation accuracy: {val_acc:.4f}")
        print(f"validation precision: {val_prec:.4f}")
        print(f"validation recall: {val_rec:.4f}")
        # test
        test_logits = model(X_test)
        test_pred = test_logits.argmax(dim=1)
        test_acc = accuracy_score(y_test.cpu(), test_pred.cpu())
        test_prec = precision_score(y_test.cpu(), test_pred.cpu())
        test_rec = recall_score(y_test.cpu(), test_pred.cpu())
        print(f"test accuracy: {test_acc:.4f}")
        print(f"test precision: {test_prec:.4f}")
        print(f"test recall: {test_rec:.4f}")
    print("total parameters:", pcount)
    print("estimated vram usage (float32):", f"{vram/1024:.2f} KB")
    return {
        "val_acc": val_acc, "val_prec": val_prec, "val_rec": val_rec,
        "test_acc": test_acc, "test_prec": test_prec, "test_rec": test_rec,
        "params": pcount, "vram": vram, "train_time_ms": train_time_ms
    }

# -------------------------------------
# helper for loading and splitting data (just like the llm from scratch code)
def load_and_split(loader):
    X_train, y_train, X_test, y_test = loader()
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return X_tr, y_tr, X_val, y_val, X_test, y_test

# -------------------------------------
# main: try a couple configs on both datasets
if __name__ == "__main__":
    configs = [
        {"n_hid": 32, "act": "relu"},
        {"n_hid": 32, "act": "tanh"},
        {"n_hid": 64, "act": "relu"},
    ]
    # titanic
    print("\n================ TITANIC =================")
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_and_split(load_titanic_data.load_titanic_dataset)
    for cfg in configs:
        train_and_report(X_tr, y_tr, X_val, y_val, X_test, y_test, n_hid=cfg["n_hid"], act=cfg["act"], epochs=100)
    # german credit
    print("\n================ GERMAN CREDIT =================")
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_and_split(load_german_credit_data.load_german_credit_data_dataset)
    for cfg in configs:
        train_and_report(X_tr, y_tr, X_val, y_val, X_test, y_test, n_hid=cfg["n_hid"], act=cfg["act"], epochs=100)
    print("\ndone.")
