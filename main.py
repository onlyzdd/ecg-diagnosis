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
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='models/resnet34.pth', help='Path to saved model')
    return parser.parse_args()


def train(dataloader, net, args, criterion, epoch, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss: %.4f' % running_loss)
    y_true = np.vstack(labels_list)
    y_score = np.vstack(output_list)
    aucs = cal_metric(y_true, y_score, average=None)
    avg_auc = np.mean(aucs)
    print('Avg AUC: %.4f' % avg_auc)
    

def evaluate(dataloader, net, args, criterion, device):
    print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss: %.4f' % running_loss)
    y_true = np.vstack(labels_list)
    y_score = np.vstack(output_list)
    aucs = cal_metric(y_true, y_score, average=None)
    avg_auc = np.mean(aucs)
    print('AUCs:', aucs)
    print('Avg AUC: %.4f' % avg_auc)
    if args.phase == 'train' and avg_auc > args.best_metric:
        args.best_metric = avg_auc
        torch.save(net.state_dict(), args.model_path)


if __name__ == "__main__":
    args = parse_args()
    args.best_metric = 0

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
    train_folds, val_folds, test_folds = folds[:8], folds[8:9], folds[9:]

    net = resnet34(input_channels=nleads).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    if args.phase == 'train':
        train_dataset = ECGDataset(data_dir, label_csv, train_folds, leads)
        val_dataset = ECGDataset(data_dir, label_csv, val_folds, leads)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        if args.resume:
            net.load_state_dict(torch.load(args.model_path, map_location=device))
        for epoch in range(args.epochs):
            train(train_loader, net, args, criterion, epoch, optimizer, device)
            evaluate(val_loader, net, args, criterion, device)
    else:
        test_dataset = ECGDataset(data_dir, label_csv, test_folds, leads)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        evaluate(test_loader, net, args, criterion, device)
