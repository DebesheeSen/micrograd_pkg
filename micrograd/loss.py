from .nn import Module
from .engine import Value

def softmax(logits):
    max_val = max([v.data for v in logits])
    shifted = [v - max_val for v in logits]

    exps = [v.exp() for v in shifted]

    total = Value(0.0)
    for e in exps:
        total += e

    return [e / total for e in exps]


class MSELoss(Module):
    def __call__(self, y_pred, y_true):
        if isinstance(y_pred, Value):
            return (y_pred - y_true) ** 2

        loss = Value(0.0)
        n = len(y_pred)

        for yp, yt in zip(y_pred, y_true):
            loss += (yp - yt) ** 2

        return loss / n

class CrossEntropyLoss:
    def __call__(self, logits, y_true):
        # logits: raw outputs (list of Value)
        # y_true: one-hot list of Value
        
        max_val = max(v.data for v in logits)
        shifted = [v - max_val for v in logits]

        exps = [v.exp() for v in shifted]

        total = Value(0.0)
        for e in exps:
            total += e

        log_sum_exp = total.log()

        loss = Value(0.0)
        for logit, y in zip(shifted, y_true):
            loss += -(y * (logit - log_sum_exp))

        return loss

class BinaryCrossEntropyLoss(Module):
    """Binary Cross Entropy — for binary classification with sigmoid output.
    
    Args:
        y_pred: sigmoid output Value in (0, 1)
        y_true: ground truth Value (0 or 1)
    
    Formula: -(y * log(p) + (1-y) * log(1-p))
    """
    def __call__(self, y_pred, y_true):
        if not isinstance(y_true, Value):
            y_true = Value(float(y_true))
        if not isinstance(y_pred, Value):
            y_pred = Value(float(y_pred))

        # Clamp to avoid log(0) — add tiny epsilon by offsetting data only
        eps = 1e-7
        p_clamped = Value(max(min(y_pred.data, 1 - eps), eps), _children=(y_pred,), _op="clamp")
        def backward():
            y_pred.grad += p_clamped.grad
        p_clamped._backward = backward

        loss = -(y_true * p_clamped.log() + (Value(1.0) - y_true) * (Value(1.0) - p_clamped).log())
        return loss

def accuracy(y_true, y_pred):

    # ---------- CASE 1: Binary (single Value output) ----------
    if isinstance(y_pred, Value):

        # y_true can be Value OR [Value]
        if isinstance(y_true, list):
            true_label = int(y_true[0].data)
        else:
            true_label = int(y_true.data)

        pred_label = 1 if y_pred.data > 0.5 else 0
        return 1 if pred_label == true_label else 0


    # ---------- CASE 2: Multi-class (list of Values) ----------
    if isinstance(y_pred, list):

        pred_vals = [p.data for p in y_pred]
        pred_class = pred_vals.index(max(pred_vals))

        # If y_true is one-hot
        if isinstance(y_true, list):
            true_vals = [t.data for t in y_true]
            true_class = true_vals.index(max(true_vals))
        else:
            # If y_true is class index Value
            true_class = int(y_true.data)

        return 1 if pred_class == true_class else 0