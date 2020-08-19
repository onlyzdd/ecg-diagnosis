import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
import wfdb


def prepare_input(ecg_file: str):
    if ecg_file.endswith('.mat'):
        ecg_file = ecg_file[:-4]
    ecg_data, _ = wfdb.rdsamp(ecg_file)
    nsteps, nleads = ecg_data.shape
    ecg_data = ecg_data[-5000:, :]
    result = np.zeros((5000, nleads)) # 10 s, 500 Hz
    result[-nsteps:, :] = ecg_data
    return result.T


def cal_metric(y_true, y_score, average):
    auc = roc_auc_score(y_true, y_score, average=average)
    return auc


def cal_scores(y_true, y_pred, y_score):
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    return f1, auc, acc


def find_optimal_threshold(y_true, y_score):
    thresholds = np.linspace(0, 1, 100)
    f1s = [f1_score(y_true, y_score > threshold) for threshold in thresholds]
    print(np.max(f1s))
    return thresholds[np.argmax(f1s)]