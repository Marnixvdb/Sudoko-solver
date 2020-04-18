from sklearn.metrics import recall_score as sklearn_recall_score


def recall_score(y_true, y_pred):
    return sklearn_recall_score(y_true=y_true, y_pred=y_pred["class_prediction"], average="weighted")
