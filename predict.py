import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from resnet import resnet34
from dataset import ECGDataset
from utils import cal_scores, find_optimal_threshold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Directory to data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers to load data')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use gpu')
    return parser.parse_args()


def get_thresholds(val_loader, net, device, threshold_path):
    print('Finding optimal thresholds...')
    if os.path.exists(threshold_path):
        return pickle.load(open(threshold_path, 'rb'))
    output_list, label_list = [], []
    for _, (data, label) in enumerate(tqdm(val_loader)):
        data, labels = data.to(device), label.to(device)
        output = net(data)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    thresholds = []
    for i in range(y_trues.shape[1]):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        threshold = find_optimal_threshold(y_true, y_score)
        thresholds.append(threshold)
    pickle.dump(thresholds, open(threshold_path, 'wb'))
    return thresholds


def apply_thresholds(test_loader, net, device, thresholds):
    output_list, label_list = [], []
    for _, (data, label) in enumerate(tqdm(test_loader)):
        data, labels = data.to(device), label.to(device)
        output = net(data)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    for i in range(len(thresholds)):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        y_pred = (y_score >= thresholds[i]).astype(int)
        print(cal_scores(y_true, y_pred, y_score))


if __name__ == "__main__":
    args = parse_args()
    database = os.path.basename(args.data_dir)
    args.model_path = f'models/resnet34_{database}_{args.leads}.pth'
    args.threshold_path = f'models/{database}-threshold.pkl'
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    
    if args.leads == 'all':
        leads = 'all'
        nleads = 12
    else:
        leads = args.leads.split(',')
        nleads = len(leads)
    data_dir = args.data_dir
    label_csv = os.path.join(data_dir, 'labels.csv')
    
    folds = range(1, 11)
    val_folds, test_folds = folds[8:9], folds[9:]
    
    net = resnet34(input_channels=nleads).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()

    val_dataset = ECGDataset(data_dir, label_csv, val_folds, leads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_dataset = ECGDataset(data_dir, label_csv, test_folds, leads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    thresholds = get_thresholds(val_loader, net, device, args.threshold_path)
    print('Thresholds:', thresholds)

    # print('Results on validation data:')
    # apply_thresholds(val_loader, net, device, thresholds)

    # print('Results on test data:')
    # apply_thresholds(test_loader, net, device, thresholds)
