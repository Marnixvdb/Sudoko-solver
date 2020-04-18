from sklearn.metrics import accuracy_score as sklearn_accuracy_score


def accuracy_score(y_true, y_pred):
    return sklearn_accuracy_score(y_true=y_true, y_pred=y_pred["class_prediction"])
