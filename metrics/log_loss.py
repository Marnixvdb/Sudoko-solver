from sklearn.metrics import log_loss as sklearn_log_loss


def log_loss(y_true, y_pred):
    return sklearn_log_loss(y_true=y_true, y_pred=y_pred.iloc[:, :-1])
