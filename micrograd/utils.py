import random
import math

class DataLoader:
    """Batches and optionally shuffles your dataset each epoch.
    
    Args:
        X: input features (list)
        y: targets (list)
        batch_size: number of samples per batch (default 32)
        shuffle: shuffle data each epoch (default True)
    
    Example:
        loader = DataLoader(X_train, y_train, batch_size=32)
        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                ...
    """
    def __init__(self, X, y, batch_size=32, shuffle=True):
        assert len(X) == len(y), "X and y must have the same length"
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        data = list(zip(self.X, self.y))
        if self.shuffle:
            random.shuffle(data)
        for i in range(0, len(data), self.batch_size):
            batch = data[i : i + self.batch_size]
            X_batch, y_batch = zip(*batch)
            yield list(X_batch), list(y_batch)

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)


class EarlyStopping:
    """Stop training when validation loss stops improving.
    
    Args:
        patience: epochs to wait without improvement before stopping (default 5)
        min_delta: minimum change to count as improvement (default 0.0)
        verbose: print message when triggered (default True)
    
    Example:
        early_stop = EarlyStopping(patience=5)
        for epoch in range(epochs):
            ...
            if early_stop(val_loss):
                print("Early stopping!")
                break
    """
    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')
        self.stop = False


def clip_gradients(params, max_norm=1.0):
    """Clips gradients to prevent exploding gradients.
    Scales all gradients down if their global norm exceeds max_norm.
    
    Args:
        params: list of Value parameters
        max_norm: maximum allowed gradient norm (default 1.0)
    
    Example:
        loss.backward()
        clip_gradients(model.parameters(), max_norm=1.0)
        optimizer.step()
    """
    total_norm = sum(p.grad ** 2 for p in params) ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for p in params:
            p.grad *= scale
    return total_norm  # useful for monitoring