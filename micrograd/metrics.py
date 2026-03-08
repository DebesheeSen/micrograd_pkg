"""
Classification and regression metrics.
All functions accept plain Python lists of ints/floats (not Value objects).

To convert from Value predictions:
    preds = [1 if p.data > 0.5 else 0 for p in y_preds]   # binary
    preds = [p.index(max(p)) for p in y_preds]              # multi-class
"""


def _to_labels(y):
    """Convert list of Value or list-of-Values to integer class labels."""
    from .engine import Value
    result = []
    for item in y:
        if isinstance(item, Value):
            result.append(int(round(item.data)))
        elif isinstance(item, list):
            # one-hot
            result.append(item.index(max(item, key=lambda v: v.data if isinstance(v, Value) else v)))
        else:
            result.append(int(round(item)))
    return result


def confusion_matrix(y_true, y_pred, num_classes=None):
    """Compute confusion matrix.
    
    Returns a num_classes x num_classes matrix where
    matrix[i][j] = number of samples with true class i predicted as class j.
    
    Args:
        y_true: list of true class indices (ints)
        y_pred: list of predicted class indices (ints)
        num_classes: number of classes (auto-detected if None)
    
    Example:
        cm = confusion_matrix([0,1,1,0], [0,1,0,0])
        # prints nicely with print_confusion_matrix(cm)
    """
    y_true = _to_labels(y_true)
    y_pred = _to_labels(y_pred)

    if num_classes is None:
        num_classes = max(max(y_true), max(y_pred)) + 1

    matrix = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def print_confusion_matrix(matrix):
    """Pretty-print a confusion matrix."""
    n = len(matrix)
    header = "     " + "  ".join(f"P{i}" for i in range(n))
    print(header)
    for i, row in enumerate(matrix):
        print(f"T{i}  ", "  ".join(f"{v:3d}" for v in row))


def precision(y_true, y_pred, average='macro'):
    """Precision = TP / (TP + FP)
    
    Args:
        average: 'macro' (mean per class) or 'binary' (positive class only)
    """
    y_true = _to_labels(y_true)
    y_pred = _to_labels(y_pred)
    classes = sorted(set(y_true + y_pred))

    precisions = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)

    if average == 'binary':
        return precisions[1] if len(precisions) > 1 else precisions[0]
    return sum(precisions) / len(precisions)  # macro


def recall(y_true, y_pred, average='macro'):
    """Recall = TP / (TP + FN)
    
    Args:
        average: 'macro' (mean per class) or 'binary' (positive class only)
    """
    y_true = _to_labels(y_true)
    y_pred = _to_labels(y_pred)
    classes = sorted(set(y_true + y_pred))

    recalls = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

    if average == 'binary':
        return recalls[1] if len(recalls) > 1 else recalls[0]
    return sum(recalls) / len(recalls)  # macro


def f1_score(y_true, y_pred, average='macro'):
    """F1 Score = 2 * (precision * recall) / (precision + recall)
    
    Args:
        average: 'macro' (mean per class) or 'binary' (positive class only)
    """
    y_true = _to_labels(y_true)
    y_pred = _to_labels(y_pred)
    classes = sorted(set(y_true + y_pred))

    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    if average == 'binary':
        return f1s[1] if len(f1s) > 1 else f1s[0]
    return sum(f1s) / len(f1s)  # macro


def classification_report(y_true, y_pred):
    """Print precision, recall, f1 per class and overall.
    
    Example:
        classification_report([0,1,1,0,1], [0,1,0,0,1])
    """
    y_true = _to_labels(y_true)
    y_pred = _to_labels(y_pred)
    classes = sorted(set(y_true + y_pred))

    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 50)
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = sum(1 for t in y_true if t == c)
        print(f"{c:<10} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10}")
    print("-" * 50)
    print(f"{'macro avg':<10} {precision(y_true,y_pred):>10.4f} {recall(y_true,y_pred):>10.4f} {f1_score(y_true,y_pred):>10.4f} {len(y_true):>10}")


def r2_score(y_true, y_pred):
    """R² (coefficient of determination) for regression.
    
    R² = 1 means perfect prediction.
    R² = 0 means model is as good as predicting the mean.
    R² < 0 means model is worse than predicting the mean.
    
    Args:
        y_true: list of true values (floats or Value)
        y_pred: list of predicted values (floats or Value)
    """
    from .engine import Value
    y_true = [v.data if isinstance(v, Value) else float(v) for v in y_true]
    y_pred = [v.data if isinstance(v, Value) else float(v) for v in y_pred]

    mean_true = sum(y_true) / len(y_true)
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))

    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0