import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. Load Titanic Data ---
X_train = pd.read_csv('titanic_X_train_scaled.csv').values.astype(np.float32)
X_test = pd.read_csv('titanic_X_test_scaled.csv').values.astype(np.float32)
y_train = pd.read_csv('titanic_y_train.csv').values.flatten().astype(np.int64)
y_test = pd.read_csv('titanic_y_test.csv').values.flatten().astype(np.int64)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

# --- 2. Define MLP Model ---
class TitanicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}[activation]
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

# --- 3. Parameter & RAM Counting ---
def count_params_and_ram(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ram_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Learnable parameters: {num_params}")
    print(f"Estimated RAM for parameters: {ram_bytes / 1024:.2f} KB ({ram_bytes / (1024**2):.2f} MB)")

# --- 4. Experiments ---
hidden_dims = [8, 16, 32]
activations = ['relu', 'tanh', 'sigmoid']
epochs = 30

results = []

for hidden_dim in hidden_dims:
    for activation in activations:
        print(f"\n=== Hidden nodes: {hidden_dim} | Activation: {activation} ===")
        input_dim = X_train.shape[1]
        output_dim = len(torch.unique(y_train))

        model = TitanicMLP(input_dim, hidden_dim, output_dim, activation)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        count_params_and_ram(model)

        best_test_acc = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Train accuracy
            _, preds = torch.max(outputs, 1)
            train_acc = (preds == y_train).float().mean().item()

            # Test accuracy
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test).item()
                _, test_preds = torch.max(test_outputs, 1)
                test_acc = (test_preds == y_test).float().mean().item()
            if test_acc > best_test_acc:
                best_test_acc = test_acc

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        results.append({'hidden_dim': hidden_dim, 'activation': activation, 'best_test_acc': best_test_acc})

print("\n=== Summary of Experiments ===")
for r in results:
    print(f"Hidden: {r['hidden_dim']:2d}, Activation: {r['activation']:7s} | Best Test Accuracy: {r['best_test_acc']:.4f}")

print("\nDone.")
