from sklearn.metrics import precision_score as sklearn_precision_score


def precision_score(y_true, y_pred):
    return sklearn_precision_score(y_true=y_true, y_pred=y_pred["class_prediction"], average="weighted")
