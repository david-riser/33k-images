import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def hungarian_method(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert (y_pred.size == y_true.size)
    dim = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((dim,dim), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    mappings = linear_assignment(w.max() - w)
    return mappings

def hungarian_balanced_accuracy(y_true, y_pred):
    mappings = hungarian_method(y_true, y_pred)
    
    y_pred_reassigned = y_pred.copy()
    for (source, dest) in mappings:
        indices = np.where(y_pred == source)[0]
        y_pred_reassigned[indices] = dest

    return balanced_accuracy_score(y_true, y_pred_reassigned)

def hungarian_accuracy(y_true, y_pred):
    mappings = hungarian_method(y_true, y_pred)

    y_pred_reassigned = y_pred.copy()
    for (source, dest) in mappings:
        indices = np.where(y_pred == source)[0]
        y_pred_reassigned[indices] = dest

    return accuracy_score(y_true, y_pred_reassigned)
