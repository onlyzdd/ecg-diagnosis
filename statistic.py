import pandas as pd

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Directory to dataset')
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    args = parser.parse_args()
    df = pd.read_csv(os.path.join(args.data_dir, 'labels.csv'))
    print(df[classes].sum())

