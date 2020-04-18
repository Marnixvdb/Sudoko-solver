from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
import numpy as np


def roc_auc_score(y_true, y_pred):
    # Map y_true to index values
    column_dict = {name: index for index, name in enumerate(y_pred.columns[:-1])}
    y_true = np.vectorize(column_dict.get)(y_true.values.ravel())

    return sklearn_roc_auc_score(y_true=y_true, y_score=y_pred.iloc[:, :-1], average="weighted", multi_class="ovo")
