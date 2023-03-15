import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

def cross_validate(model, X, y):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    log_losses = []
    aucs = []
    accuracies = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        log_loss_value = log_loss(y_test, y_pred_proba)
        auc_value = roc_auc_score(y_test, y_pred_proba[:, 1])
        accuracy_value = accuracy_score(y_test, y_pred)

        log_losses.append(log_loss_value)
        aucs.append(auc_value)
        accuracies.append(accuracy_value)

    mean_log_loss = np.mean(log_losses)
    mean_auc = np.mean(aucs)
    mean_accuracy = np.mean(accuracies)

    return mean_log_loss, mean_auc, mean_accuracy
