import pandas as pd

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Directory to dataset')
    args = parser.parse_args()
    
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    df_labels = pd.read_csv(os.path.join(args.data_dir, 'labels.csv'))
    df_reference = pd.read_csv(os.path.join(args.data_dir, 'reference.csv'))
    
    df = pd.merge(df_labels, df_reference[['patient_id', 'age', 'sex', 'signal_len']], on='patient_id', how='left')
    df['sex'] = (df['sex'] == 'Male').astype(int)
    df['signal_len'] = df['signal_len'] / 500

    N = len(df)

    results = []

    for col in classes:
        result = []
        df_tmp = df[df[col] == 1]
        result.append('%d (%.2f)' % (len(df_tmp), len(df_tmp) * 100 / N)) # count
        result.append('%d (%.2f)' % (df_tmp['sex'].sum(), df_tmp['sex'].mean() * 100)) # count and percentage of males
        result.append('%.2f (%.2f)' % (df_tmp['age'].mean(), df_tmp['age'].std())) # mean and std age
        result.append('%.2f (%.2f)' % (df_tmp['signal_len'].mean(), df_tmp['signal_len'].std())) # mean and std length
        results.append(result)
    
    df_stat = pd.DataFrame(results, index=classes, columns=['N', 'Male (%)', 'Age', 'Signal length'])
    print(df_stat) # the statistics is different from official, may because they only use first label
