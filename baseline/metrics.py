import numpy as np

from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import confusion_matrix

def hungarian_method(y_true, y_pred):

    y_true = y_true.astype(np.int64)
    assert (y_pred.size == y_true.size)
    dim = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((dim,dim), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    return ind

def hungarian_method2(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    return linear_assignment(confusion.max() - confusion)
