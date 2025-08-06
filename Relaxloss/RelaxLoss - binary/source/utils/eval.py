from __future__ import print_function, absolute_import
import numpy as np
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import metrics

__all__ = ['accuracy', 'accuracy_binary', 'metrics_binary', 'plot_roc']


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if output.ndim == 1 or output.size(1) == 1:
        # Binary case: sigmoid output expected
        pred = (output >= 0.5).long()
        correct = pred.eq(target.view_as(pred)).sum().item()
        return [100.0 * correct / target.size(0)] + [0.0 for _ in range(len(topk)-1)]

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def accuracy_binary(output, target):
    """Binary classification accuracy, input and target are expected as float tensors."""
    pred = (output >= 0.5).float()
    correct = pred.eq(target).sum().item()
    return 100.0 * correct / target.size(0)


def metrics_binary(y_true, y_score):
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    f1 = f1_score(y_true, (y_score >= 0.5).astype(int))
    pos_num = y_true.sum()
    frac = pos_num / len(y_true)
    return auc, ap, f1, pos_num, frac



def plot_roc(pos_results, neg_results):
    labels = np.concatenate((np.zeros((len(neg_results),)), np.ones((len(pos_results),))))
    results = np.concatenate((neg_results, pos_results))
    fpr, tpr, threshold = metrics.roc_curve(labels, results, pos_label=1)
    auc = metrics.roc_auc_score(labels, results)
    ap = metrics.average_precision_score(labels, results)
    return fpr, tpr, threshold, auc, ap
