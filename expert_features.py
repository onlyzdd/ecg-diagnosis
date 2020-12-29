from biosppy import tools
import numpy as np
import pandas as pd
import pywt
from scipy.stats import entropy, kurtosis, skew

from QRSDetectorOffline import QRSDetectorOffline


def cal_entropy(coeff):
    # calculate shannon entropy
    coeff = pd.Series(coeff).value_counts()
    e = entropy(coeff)
    return e / 10


def extract_stats(channel):
    # extract statistic features
    n5 = np.percentile(channel, 5)
    n25 = np.percentile(channel, 25)
    n75 = np.percentile(channel, 75)
    n95 = np.percentile(channel, 95)
    median = np.percentile(channel, 50)
    mean = np.mean(channel)
    std = np.std(channel)
    return [n5, n25, n75, n95, median, mean, std]


def extract_wavelet_features(channel):
    # extract wavelet coeff features
    coeffs = pywt.wavedec(channel, 'db1', level=3)
    return coeffs[0] + [cal_entropy(coeffs[0])]


def extract_heart_rates(ecg_data, sampling_rate=500):
    # extract instant heart rates
    qrs_detector = QRSDetectorOffline(ecg_data, frequency=sampling_rate)
    _, heart_rates = tools.get_heart_rate(qrs_detector.detected_peaks_indices, sampling_rate=sampling_rate)
    return extract_stats(heart_rates / 100)


def extract_hos(channel):
    # extract higher order statistics features
    return []


def extract_channel_features(channel):
    stats_features = extract_stats(channel)
    wavelet_features = extract_wavelet_features(channel)
    hos_features = extract_hos(channel)
    return stats_features + wavelet_features + hos_features

   
def extract_features(ecg_data, sampling_rate=500):
    # extract expert features for 12-lead ECGs
    all_features = []
    all_features += extract_heart_rates(ecg_data, sampling_rate=sampling_rate)
    for channel in ecg_data.T:
        all_features += extract_channel_features(channel)
    return all_features
