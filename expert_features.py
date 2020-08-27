import pywt

from biosppy import ecg, tools
import numpy as np
import pandas as pd
import scipy


def cal_entropy(coeff):
    coeff = pd.Series(coeff).value_counts()
    entropy = scipy.stats.entropy(coeff)
    return entropy / 10


def cal_statistics(signal):
    n5 = np.percentile(signal, 5)
    n25 = np.percentile(signal, 25)
    n75 = np.percentile(signal, 75)
    n95 = np.percentile(signal, 95)
    median = np.percentile(signal, 50)
    mean = np.mean(signal)
    std = np.std(signal)
    var = np.var(signal)
    return [n5, n25, n75, n95, median, mean, std, var]


def extract_lead_heart_rate(signal, sampling_rate):
    # extract heart rate for single-lead ECG: may return empty list
    rpeaks, = ecg.hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)
    rpeaks, = ecg.correct_rpeaks(signal=signal, rpeaks=rpeaks, sampling_rate=sampling_rate, tol=0.05)
    _, heartrates = tools.get_heart_rate(beats=rpeaks, sampling_rate=500, smooth=True, size=3)
    return list(heartrates / 100) # divided by 100


def extract_heart_rates(ecg_data, sampling_rate=500):
    # extract heart rates using 12-lead since rpeaks can not be detected on some leads
    heartrates = []
    for signal in ecg_data.T:
        lead_heartrates = extract_lead_heart_rate(signal=signal, sampling_rate=sampling_rate)
        heartrates += lead_heartrates
    return cal_statistics(heartrates)


def extract_lead_features(signal):
    # extract expert features for single-lead ECGs: statistics, shannon entropy
    lead_features = cal_statistics(signal) # statistic of signal
    coeffs = pywt.wavedec(signal, 'db10', level=4)
    for coeff in coeffs:
        lead_features.append(cal_entropy(coeff)) # shannon entropy of coefficients
        lead_features += cal_statistics(coeff) # statistics of coefficients
    return lead_features

   
def extract_features(ecg_data, sampling_rate=500):
    # extract expert features for 12-lead ECGs
    # may include heart rates later
    all_features = []
    # comment out below line to extract heart rates
    # all_features += extract_heart_rates(ecg_data, sampling_rate=sampling_rate)
    for signal in ecg_data.T:
        all_features += extract_lead_features(signal)
    return all_features
