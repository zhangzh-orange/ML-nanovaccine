# -*- coding: utf-8 -*-
# @Time    : 2021/11/1 0001 15:09
# @Author  : zhangzh_orange
# @FileName: metrics.py

import numpy as np
import sklearn.metrics as m
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from numba import njit


@njit
def c_index(y_true, y_pred):
    summ = 0
    pair = 0

    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1

    if pair != 0:
        return summ / pair
    else:
        return 0


def RMSE(y_true, y_pred):
    return np.sqrt(m.mean_squared_error(y_true, y_pred))


def MAE(y_true, y_pred):
    return m.mean_absolute_error(y_true, y_pred)


def CORR(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def SD(y_true, y_pred):
    from sklearn.linear_model import LinearRegression
    y_pred = y_pred.reshape((-1, 1))
    lr = LinearRegression().fit(y_pred, y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))


def SP(y_true, y_pred):
    """Spearmanr"""
    return spearmanr(y_true, y_pred)[0]


def macro_P(y_true, y_pred):
    """Micro precision"""
    return m.precision_score(y_true, y_pred, average="macro")


def macro_R(y_true, y_pred):
    """Micro recall"""
    return m.recall_score(y_true, y_pred, average='macro')


def micro_f1(y_true, y_pred):
    """Micro f1-score"""
    return m.f1_score(y_true, y_pred, average='micro')


def macro_f1(y_true, y_pred):
    """Macro f1-score"""
    return m.f1_score(y_true, y_pred, average='macro')


def macro_AUC(y_true, y_pred):
    """Macro AUC"""
    return m.roc_auc_score(y_true, y_pred, multi_class='ovo', average='macro')


def AUC(y_true, y_pred):
    """Binary precision"""
    return m.roc_auc_score(y_true, y_pred, multi_class='ovr')


def kappa(y_true, y_pred):
    """Cohen’s kappa"""
    return m.cohen_kappa_score(y_true, y_pred)

