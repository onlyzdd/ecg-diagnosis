import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from dataset import ECGDataset
from resnet import resnet34
from utils import cal_metric


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Directory for data dir')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='models/resnet34.pth', help='Path to saved model')
    return parser.parse_args()

args = parse_args()
if args.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = 'cpu'
best_metric = 0

def train(dataloader, net, criterion, epoch, optimizer):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    aucs = []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss: %.3f' % running_loss)
    y_trues = np.concatenate(labels_list).T
    y_probs = np.concatenate(output_list).T
    for i in range(y_trues.shape[0]):
        auc = cal_metric(y_trues[i], y_probs[i])
        aucs.append(auc)
    average_auc = np.mean(aucs)
    print('AUCs', ' '.join([str(auc) for auc in aucs]))
    print('Avg AUC: ', average_auc)


def evaluate(dataloader, net, criterion):
    global best_metric
    print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    aucs = []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss: %.3f' % running_loss)
    y_trues = np.concatenate(labels_list).T
    y_probs = np.concatenate(output_list).T
    for i in range(y_trues.shape[0]):
        if np.std(y_trues[i]) == 0:
            auc = 0
        else:
            auc = cal_metric(y_trues[i], y_probs[i])
        aucs.append(auc)
    average_auc = np.mean(aucs)
    print('AUCs', ' '.join([str(auc) for auc in aucs]))
    print('Avg AUC: ', average_auc)
    if average_auc > best_metric:
        best_metric = average_auc
        torch.save(net.state_dict(), args.model_path)


if __name__ == "__main__":
    data_dir = args.data_dir
    database = os.path.basename(data_dir)
    label_csv = os.path.join(data_dir, 'labels.csv')
    folds = range(1, 11)
    train_folds, val_folds, test_folds = folds[:8], folds[8:9], folds[9:]
    train_dataset = ECGDataset(data_dir, label_csv, train_folds)
    val_dataset = ECGDataset(data_dir, label_csv, val_folds)
    test_dataset = ECGDataset(data_dir, label_csv, test_folds)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    net = resnet34(input_channels=12).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    if args.phase == 'train':
        for epoch in range(args.epochs):
            train(train_loader, net, criterion, epoch, optimizer)
            evaluate(val_loader, net, criterion)
    else:
        net.load_state_dict(torch.load(args.model_path))
        evaluate(test_loader, net, criterion)
