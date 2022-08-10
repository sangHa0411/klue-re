
import numpy as np
from sklearn.metrics import (accuracy_score, 
    precision_recall_curve,
    f1_score,
    auc
)

def auprc_score(probs, labels):
    labels = np.eye(30)[labels]
    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = precision_recall_curve(targets_c, preds_c)
        score[c] = auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(predictions, labels):
    preds = predictions.argmax(-1)
    probs = np.exp(predictions) / np.sum(np.exp(predictions))

    label_indices = list(range(30))
    f1 = f1_score(labels, preds, average="micro", labels=label_indices) * 100.0
    auprc = auprc_score(probs, labels)
    acc = accuracy_score(labels, preds)
    return {
        'f1': f1,
        'auprc' : auprc,
        'accuracy': acc,
    }