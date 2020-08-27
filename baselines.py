import argparse
import os
import warnings

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm
import wfdb

from utils import split_data, find_optimal_threshold
from expert_features import extract_features

warnings.filterwarnings('ignore', category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Data directory')
    parser.add_argument('--classifier', type=str, default='all', help='Classifier to use: LR, RF, LGB, or MLP')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    return parser.parse_args()


def generate_features_csv(features_csv, data_dir, patient_ids):
    print('Generating expert features...')
    ecg_features = []
    for patient_id in tqdm(patient_ids):
        ecg_data, _ = wfdb.rdsamp(os.path.join(data_dir, patient_id))
        ecg_features.append(extract_features(ecg_data))
    df = pd.DataFrame(ecg_features, index=patient_ids)
    df.index.name = 'patient_id'
    df.to_csv(features_csv)
    return df


if __name__ == "__main__":
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    args = parse_args()
    data_dir = args.data_dir
    classifier = args.classifier
    features_csv = os.path.join(data_dir, 'features.csv')
    labels_csv = os.path.join(data_dir, 'labels.csv')

    df_labels = pd.read_csv(labels_csv)
    patient_ids = df_labels['patient_id'].tolist()
    if not os.path.exists(features_csv):
        df_X = generate_features_csv(features_csv, data_dir, patient_ids)
    else:
        df_X = pd.read_csv(features_csv)
    df_X = df_X.merge(df_labels[['patient_id', 'fold']], on='patient_id')
    
    train_folds, val_folds, test_folds = split_data(seed=args.seed)
    feature_cols = df_X.columns[1:-1] # remove patient id and fold

    X_train = df_X[df_X['fold'].isin(train_folds)][feature_cols].to_numpy()
    X_val = df_X[df_X['fold'].isin(val_folds)][feature_cols].to_numpy()
    X_test = df_X[df_X['fold'].isin(test_folds)][feature_cols].to_numpy()

    y_train = df_labels[df_labels['fold'].isin(train_folds)][classes].to_numpy()
    y_val = df_labels[df_labels['fold'].isin(val_folds)][classes].to_numpy()
    y_test = df_labels[df_labels['fold'].isin(test_folds)][classes].to_numpy()

    if classifier == 'all':
        classifiers = ['LR', 'RF', 'LGB', 'MLP']
    else:
        classifiers = [classifier]

    for classifier in classifiers:
        # tune parameters
        if classifier == 'LR':
            model = LogisticRegression(solver='lbfgs', max_iter=1000)
        elif classifier == 'RF':
            model = RandomForestClassifier(n_estimators=300, max_depth=10)
        elif classifier == 'LGB':
            model = LGBMClassifier(n_estimators=100)
        else:
            model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
        if classifier != 'MLP':
            model = OneVsRestClassifier(model)

        print(f'Start training {classifier}...')
        model.fit(X_train, y_train)
        
        y_val_scores = model.predict_proba(X_val)
        y_test_scores = model.predict_proba(X_test)
        
        f1s = []
        thresholds = []
        print('Finding optimal thresholds on validation dataset...')

        for i in range(len(classes)):
            # find optimal threshold on validation dataset
            y_val_score = y_val_scores[:, i]
            threshold = find_optimal_threshold(y_val[:, i], y_val_score)
            # apply optimal threshold to test dataset
            y_test_score = y_test_scores[:, i]
            y_test_pred = y_test_score > threshold
            f1 = f1_score(y_test[:, i], y_test_pred)
            thresholds.append(threshold)
            f1s.append(f1)
        np.set_printoptions(precision=3)
        print(f'{classifier} F1s:', f1s)
        print('Avg F1:', np.mean(f1s))
