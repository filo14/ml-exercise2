"""Minimal pure‑NumPy feed‑forward neural network with grid‑search.
Each hidden layer uses the same activation (default: ReLU);
The final layer is Softmax and the loss function is
cross‑entropy operating *on the soft‑max probabilities*.

Run this file directly to reproduce the Titanic and German‑Credit
experiments from the exercise – no external deep‑learning libraries
required, only NumPy, pandas and scikit‑learn.
"""
from __future__ import annotations

import itertools
import time
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

import load_german_credit_data
import load_titanic_data

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
DTYPE = np.float32  # switch to float64 for extra precision

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.size, num_classes), dtype=DTYPE)
    out[np.arange(y.size), y] = 1.0
    return out

# -----------------------------------------------------------------------------
# Activation base class and concrete activations
# -----------------------------------------------------------------------------

class Activation:
    """Interface for activations (must implement forward/backward)."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0  # boolean mask – saves memory vs. storing x itself
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask


class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-x, dtype=DTYPE))
        self._out = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        s = self._out
        return grad_out * s * (1.0 - s)


class Softmax(Activation):
    """Softmax layer – kept separate so the user sees an explicit softmax."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shifted, dtype=DTYPE)
        self._out = exps / np.sum(exps, axis=1, keepdims=True)
        return self._out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # Jacobian‑vector product for softmax
        s = self._out
        dot = np.sum(grad_out * s, axis=1, keepdims=True)
        return s * (grad_out - dot)

# -----------------------------------------------------------------------------
# Dense / fully connected layer
# -----------------------------------------------------------------------------

class Dense:
    def __init__(self, in_features: int, out_features: int, *, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = rng.normal(0.0, scale, size=(in_features, out_features)).astype(DTYPE)
        self.b = np.zeros(out_features, dtype=DTYPE)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x  # cache for backward
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._x
        self.dW[...] = (x.T @ grad_out) / x.shape[0]
        self.db[...] = grad_out.mean(axis=0)
        return grad_out @ self.W.T

    # expose params / grads for optimisers
    @property
    def params(self):
        return [self.W, self.b]

    @property
    def grads(self):
        return [self.dW, self.db]

# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

class Loss:  # base
    def forward(self, y_pred, y_true):  # pragma: no cover
        raise NotImplementedError

    def backward(self, y_pred, y_true):  # pragma: no cover
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """Cross‑entropy that EXPECTS *probabilities* (softmax outputs)."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-9
        probs = np.clip(y_pred, eps, 1.0)
        loss = -np.log(probs[np.arange(y_true.size), y_true]).mean()
        self._probs = probs
        self._y_true = y_true
        return loss

    def backward(self, *_):  # returns dL/dp where p are probabilities
        grad = self._probs.copy()
        grad[np.arange(self._y_true.size), self._y_true] -= 1.0
        return grad / self._y_true.size

# -----------------------------------------------------------------------------
# Optimiser – SGD + optional momentum
# -----------------------------------------------------------------------------

class SGD:
    def __init__(self, params: List[np.ndarray], grads: List[np.ndarray], *, lr: float = 1e-2, momentum: float = 0.0):
        self.params, self.grads = params, grads
        self.lr, self.momentum = lr, momentum
        self._velocity = [np.zeros_like(p) for p in params]

    def step(self):
        for p, g, v in zip(self.params, self.grads, self._velocity):
            v[...] = self.momentum * v + g
            p[...] -= self.lr * v

    def zero_grad(self):
        for g in self.grads:
            g[...] = 0.0

# -----------------------------------------------------------------------------
# Neural‑network wrapper
# -----------------------------------------------------------------------------

class NeuralNetwork:
    def __init__(
        self,
        layer_sizes: Sequence[int],
        *,
        activation: str | Activation = "relu",
        num_classes: int,
        lr: float = 1e-2,
        momentum: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        # map string -> activation class
        if isinstance(activation, str):
            act_map = {"relu": ReLU, "sigmoid": Sigmoid}
            hidden_act_cls = act_map[activation.lower()]
        else:
            hidden_act_cls = type(activation)

        self._hidden_act_cls = hidden_act_cls
        self._layer_sizes = list(layer_sizes)
        self._num_classes = num_classes
        self._rng = rng or np.random.default_rng()

        # will be built lazily once input dimension known
        self._deferred = True
        self.layers: List[Dense | Activation] = []
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = SGD([], [], lr=lr, momentum=momentum)  # placeholder, replaced after build

    # ------------------------------------------------------------------ helpers
    def _materialise_layers(self, in_features: int):
        if not self._deferred:
            return
        prev = in_features
        layers: List[Dense | Activation] = []
        for hid in self._layer_sizes:
            layers.append(Dense(prev, hid, rng=self._rng))
            layers.append(self._hidden_act_cls())  # *new* instance per layer!
            prev = hid
        # output layer + softmax
        layers.append(Dense(prev, self._num_classes, rng=self._rng))
        layers.append(Softmax())

        # expose parameters to optimiser
        params, grads = [], []
        for l in layers:
            if isinstance(l, Dense):
                params.extend(l.params)
                grads.extend(l.grads)
        self.optimizer.params[:] = params
        self.optimizer.grads[:] = grads
        self.optimizer._velocity = [np.zeros_like(p) for p in params]

        self.layers = layers
        self._deferred = False

    # ------------------------------------------------------------ forward pass
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self._deferred:
            self._materialise_layers(x.shape[1])
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # ----------------------------------------------------------- backward pass
    def backward(self, loss_grad: np.ndarray):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # ----------------------------------------------------------------- training
    def fit(self, X, y, *, epochs: int = 100, batch_size: int | None = None, X_val=None, y_val=None, verbose: bool = True):
        N = X.shape[0]
        batch_size = batch_size or N
        history = {"loss": [], "val_loss": []}
        for ep in range(epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, batch_size):
                xb = X[idx[start : start + batch_size]]
                yb = y[idx[start : start + batch_size]]

                probs = self.forward(xb)
                loss = self.loss_fn.forward(probs, yb)
                grad = self.loss_fn.backward(probs, yb)
                self.backward(grad)
                self.optimizer.step()
                self.optimizer.zero_grad()

            history["loss"].append(loss)
            if X_val is not None:
                v_probs = self.forward(X_val)
                v_loss = self.loss_fn.forward(v_probs, y_val)
                history["val_loss"].append(v_loss)

            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                msg = f"Epoch {ep + 1}/{epochs}: loss={history['loss'][-1]:.4f}"
                if X_val is not None:
                    msg += f", val_loss={history['val_loss'][-1]:.4f}"
                print(msg)
        return history

    # --------------------------------------------------------------- utilities
    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def parameter_count(self):
        return sum(p.size for p in self.optimizer.params)

    
    def vram_usage(self):
        total_bytes = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                weight_bytes = layer.W.size * layer.W.dtype.itemsize
                bias_bytes = layer.b.size * layer.b.dtype.itemsize
                total_bytes += weight_bytes + bias_bytes

        return total_bytes

# -----------------------------------------------------------------------------
# Grid‑search helper
# -----------------------------------------------------------------------------

def grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: Dict[str, Iterable],
    *,
    num_classes: int,
    # max_epochs: int = 1000, # Removed this parameter
):
    keys = list(param_grid)
    best_acc, best_model, best_cfg, best_runtime = -np.inf, None, {}, np.inf
    for values in itertools.product(*param_grid.values()):
        start_time = time.time()
        cfg = dict(zip(keys, values))
        model = NeuralNetwork(
            layer_sizes=cfg["layer_sizes"],
            activation=cfg.get("activation", "relu"),
            num_classes=num_classes,
            lr=cfg.get("lr", 1e-2),
            momentum=cfg.get("momentum", 0.0),
        )
        # Use epochs from the current configuration (cfg)
        epochs_for_fit = cfg.get("epochs", 1000) # Default to 1000 if not in cfg
        model.fit(X_train, y_train, epochs=epochs_for_fit, verbose=False)
        acc = model.score(X_val, y_val)
        if acc > best_acc:
            best_acc, best_model, best_cfg = acc, model, cfg
            print(f"New best val acc {acc:.4f} with cfg {cfg}")
            end_time = time.time()
            best_runtime = f"{((end_time - start_time)* 1000):.3f}"
    return best_model, best_cfg, best_runtime

# -----------------------------------------------------------------------------
# Data‑loader – identical to the snippet provided by the exercise sheet
# -----------------------------------------------------------------------------

def load_titanic_dataset():
    """Return train/test splits (already scaled) for the Kaggle Titanic dataset."""
    X_train_df = pd.read_csv("./titanic-preprocessing/titanic_X_train_scaled.csv")
    y_train_df = pd.read_csv("./titanic-preprocessing/titanic_y_train.csv")
    X_test_df = pd.read_csv("./titanic-preprocessing/titanic_X_test_scaled.csv")
    y_test_df = pd.read_csv("./titanic-preprocessing/titanic_y_test.csv")

    X_train = X_train_df.to_numpy(dtype=DTYPE)
    y_train = y_train_df["Survived"].to_numpy(dtype=np.int32).reshape(-1)

    X_test = X_test_df.to_numpy(dtype=DTYPE)
    y_test = y_test_df["Survived"].to_numpy(dtype=np.int32).reshape(-1)

    return X_train, y_train, X_test, y_test

# -----------------------------------------------------------------------------
# Experiment helper
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Experiment helper
# -----------------------------------------------------------------------------

def run_experiment(
    name: str,
    loader_fn: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    seed: int = 42,
):
    print(f"\n\n========== {name} ==========")

    # 1) ---------------------------------------------------------------- data
    X_train, y_train, X_test, y_test = loader_fn()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=seed,
    )

    # 2) ----------------------------------------------------------- grid search
    param_grid = {
        "layer_sizes": [[32],
    [64, 32],
    [128, 64, 32],
    [256, 128, 64],
    [64, 64, 64]],
        "activation": ["relu", "sigmoid"],
        "lr": [1],
        "momentum": [0.0],
        "epochs": [100, 500, 1000], # Add epochs as an array here
    }

    start = time.perf_counter()
    best_model, best_cfg, runtime = grid_search(
        X_tr,
        y_tr,
        X_val,
        y_val,
        param_grid,
        num_classes=2 if name == "Titanic" else 2,
        # max_epochs=500, # Remove this line
    )
    gsearch_time = (time.perf_counter() - start) * 1_000
    print(f"Grid‑search done in {gsearch_time:,.0f} ms")
    print("Best configuration:", best_cfg)

    # 3) ------------------------------------------------------------- evaluate
    print("\nValidation accuracy:", best_model.score(X_val, y_val))
    print("Test accuracy      :", best_model.score(X_test, y_test))
    print("Total parameters   :", best_model.parameter_count())
    print("VRAM (float32)     :", f"{best_model.vram_usage():.2f} bytes")
    print("Runtime:            ", runtime)

    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)
    print(
        f"Validation precision / recall: {precision_score(y_val, y_val_pred):.4f} / {recall_score(y_val, y_val_pred):.4f}"
    )
    print(
        f"Test precision / recall      : {precision_score(y_test, y_test_pred):.4f} / {recall_score(y_test, y_test_pred):.4f}"
    )


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main():
    run_experiment("Titanic", load_titanic_data.load_titanic_dataset)
    run_experiment("German Credit", load_german_credit_data.load_german_credit_data_dataset)


if __name__ == "__main__":
    main()
