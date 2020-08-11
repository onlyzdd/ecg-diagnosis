import argparse
from glob import glob
import os

import numpy as np
import torch
import shap

from resnet import resnet34
from utils import prepare_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', type=str, default='data/CPSC', help='Data directory')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='models/resnet34.pth')
    args = parser.parse_args()
    inputs = glob(os.path.join(args.input_dir, 'A001*.mat'))
    inputs = torch.stack([torch.from_numpy(prepare_input(input0)).float() for input0 in inputs])
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    model = resnet34(input_channels=12).to(device)
    model.load_state_dict(torch.load(args.model_path))
    e = shap.GradientExplainer(model, inputs)
    sv = e.shap_values(inputs)
    np.save('results/A001.npy', sv)
