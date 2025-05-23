from __future__ import annotations
import itertools
from typing import Callable, List, Dict, Tuple, Iterable, Sequence, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DTYPE = np.float32  # switch to float64 for extra precision

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.size, num_classes), dtype=DTYPE)
    out[np.arange(y.size), y] = 1.0
    return out

# -----------------------------------------------------------------------------
# Activations
# -----------------------------------------------------------------------------

class Activation:
    def __call__(self, x: np.ndarray) -> np.ndarray:  # alias for forward
        return self.forward(x)
    def forward(self, x: np.ndarray) -> np.ndarray:  # noqa: D401
        raise NotImplementedError
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0  # bool mask saves RAM over raw tensor
        return x * self._mask
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask

class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-x))
        self._cache = out
        return out
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        s = self._cache
        return grad_out * s * (1.0 - s)

class Tanh(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(x)
        self._cache = out
        return out
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        t = self._cache
        return grad_out * (1.0 - t ** 2)

# -----------------------------------------------------------------------------
# Dense layer
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
        self._x = x  # cache input for grad
        return x @ self.W + self.b
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._x
        self.dW[...] = x.T @ grad_out / x.shape[0]
        self.db[...] = grad_out.mean(axis=0)
        return grad_out @ self.W.T
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
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    def backward(self, y_pred, y_true):
        raise NotImplementedError

class CrossEntropyLoss(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # shift for stability
        logits = y_pred - y_pred.max(axis=1, keepdims=True)
        self._probs = np.exp(logits)
        self._probs /= self._probs.sum(axis=1, keepdims=True)
        loss = -np.log(self._probs[np.arange(y_true.size), y_true]).mean()
        self._y_true = y_true
        return loss
    def backward(self, *_):
        grad = self._probs.copy()
        grad[np.arange(self._y_true.size), self._y_true] -= 1.0
        return grad / self._y_true.size

class MSELoss(Loss):
    def forward(self, y_pred, y_true):
        self._diff = y_pred - y_true
        return np.square(self._diff).mean()
    def backward(self, *_):
        return 2.0 * self._diff / self._diff.shape[0]

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------

class SGD:
    def __init__(self, params: List[np.ndarray], grads: List[np.ndarray], lr: float = 1e-2, momentum: float = 0.0):
        self.params, self.grads = params, grads
        self.lr, self.momentum = lr, momentum
        self._velocity = [np.zeros_like(p) for p in params]
    def step(self):
        for p, g, v in zip(self.params, self.grads, self._velocity):
            v[...] = self.momentum * v + g  # correct update rule
            p[...] -= self.lr * v
    def zero_grad(self):
        for g in self.grads:
            g[...] = 0.0

# -----------------------------------------------------------------------------
# NeuralNetwork wrapper
# -----------------------------------------------------------------------------

class NeuralNetwork:
    def __init__(
        self,
        layer_sizes: Sequence[int],
        activations: Sequence[str | Activation],
        num_classes: int,
        *,
        loss: str | Loss = "cross_entropy",
        lr: float = 1e-2,
        momentum: float = 0.0,
        rng: np.random.Generator | None = None,
    ):
        # map string -> activation objects
        if isinstance(activations[0], str):
            act_map = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}
            acts = [act_map[a.lower()]() for a in activations]
        else:
            acts = list(activations)
        assert len(acts) == len(layer_sizes), "Need one activation per hidden layer"

        # store config and mark for deferred build
        self._cfg = (layer_sizes, acts, num_classes, rng)
        self._deferred = True
        self.layers: List[Dense | Activation] = []

        # create dummy optimizer so attributes exist; will be replaced later
        self.optimizer = SGD([], [], lr=lr, momentum=momentum)

        self.loss_fn = CrossEntropyLoss() if loss == "cross_entropy" else MSELoss() if isinstance(loss, str) else loss

    # ---------------------------------------------------------------
    def _materialise_layers(self, in_features: int):
        if not self._deferred:
            return
        layer_sizes, acts, num_classes, rng = self._cfg
        prev = in_features
        layers: List[Dense | Activation] = []
        for hid, act in zip(layer_sizes, acts):
            layers.append(Dense(prev, hid, rng=rng))
            layers.append(act)
            prev = hid
        layers.append(Dense(prev, num_classes, rng=rng))
        self.layers = layers
        # rebuild optimizer parameter lists
        params, grads = [], []
        for l in layers:
            if isinstance(l, Dense):
                params.extend(l.params)
                grads.extend(l.grads)
        self.optimizer.params[:] = params
        self.optimizer.grads[:] = grads
        self.optimizer._velocity = [np.zeros_like(p) for p in params]  # reset velocity
        self._deferred = False

    # ---------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self._deferred:
            self._materialise_layers(x.shape[1])
        out = x
        for layer in self.layers:
            out = layer.forward(out) if hasattr(layer, "forward") else layer(out)
        return out

    # ---------------------------------------------------------------
    def backward(self, loss_grad: np.ndarray):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # ---------------------------------------------------------------
    def fit(self, X, y, *, epochs=100, batch_size=None, X_val=None, y_val=None, verbose=True):
        N = X.shape[0]
        batch_size = batch_size or N
        history = {"loss": [], "val_loss": []}
        for ep in range(epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, batch_size):
                xb, yb = X[idx[start:start + batch_size]], y[idx[start:start + batch_size]]
                logits = self.forward(xb)
                loss = self.loss_fn.forward(logits, yb)
                grad = self.loss_fn.backward(logits, yb)
                self.backward(grad)
                self.optimizer.step()
                self.optimizer.zero_grad()
            history["loss"].append(loss)
            if X_val is not None:
                v_logits = self.forward(X_val)
                v_loss = self.loss_fn.forward(v_logits, y_val)
                history["val_loss"].append(v_loss)
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                msg = f"Epoch {ep + 1}/{epochs}: loss={history['loss'][-1]:.4f}"
                if X_val is not None:
                    msg += f", val_loss={history['val_loss'][-1]:.4f}"
                print(msg)
        return history

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def parameter_count(self):
        return sum(p.size for p in self.optimizer.params)

    def vram_usage(self, bytes_per_param=4):
        return self.parameter_count() * bytes_per_param / 1024 ** 2

# -----------------------------------------------------------------------------
# Grid search convenience
# -----------------------------------------------------------------------------

def grid_search(X_train, y_train, X_val, y_val, param_grid: Dict[str, Iterable], *, num_classes, max_epochs=100):
    keys = list(param_grid)
    best_acc, best_model, best_cfg = -np.inf, None, {}
    for values in itertools.product(*param_grid.values()):
        cfg = dict(zip(keys, values))
        model = NeuralNetwork(layer_sizes=cfg["layer_sizes"], activations=cfg["activations"], num_classes=num_classes, lr=cfg.get("lr", 1e-2), momentum=cfg.get("momentum", 0.0))
        model.fit(X_train, y_train, epochs=max_epochs, verbose=False)
        acc = model.score(X_val, y_val)
        if acc > best_acc:
            best_acc, best_model, best_cfg = acc, model, cfg
            print(f"New best val acc {acc:.4f} with cfg {cfg}")
    return best_model, best_cfg



# -----------------------------------------------------------------------------
# Data loader – identical to the snippet provided by the exercise sheet
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
# Main experiment
# -----------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------ data
    X_train, y_train, X_test, y_test = load_titanic_dataset()

    # 20‑% of the official training portion becomes a validation hold‑out
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    # --------------------------------------------------------- model & train
    # Two hidden layers (64 → 32) with ReLU activations proved a good
    # trade‑off between capacity and over‑fitting in quick grid‑search pilots.
    model = NeuralNetwork(
        layer_sizes=[64, 32],
        activations=["relu", "relu"],
        num_classes=2,
        lr=1e-2,
        momentum=0.9,
    )

    history = model.fit(
        X_tr,
        y_tr,
        epochs=2000,
        batch_size=32,
        X_val=X_val,
        y_val=y_val,
        verbose=True,
    )

    # -------------------------------------------------------------- evaluate
    print("\nValidation accuracy:", model.score(X_val, y_val))
    print("Test accuracy:", model.score(X_test, y_test))
    print("Total parameters:", model.parameter_count())
    print("Estimated VRAM usage (float32):", f"{model.vram_usage():.2f} MB")

    # Detailed report for the submission slides
    y_pred = model.predict(X_test)
    print("\nClassification report (Test set):\n", classification_report(y_test, y_pred, target_names=["Not survived", "Survived"]))



if __name__ == "__main__":
    main()
