# Interpretable Deep Learning for Automatic Diagnosis of 12-lead Electrocardiogram

This repository contains code for *Interpretable Deep Learning for Automatic Diagnosis of 12-lead Electrocardiogram*. Electrocardiogram (ECG) is a widely used reliable, non-invasive approach for cardiovascular disease diagnosis. With the rapid growth of ECG examinations and the insufficiency of cardiologists, accurately automatic diagnosis of ECG signals has become a hot research topic. Deep learning methods have demonstrated promising results in predictive healthcare tasks. In this work, we developed a deep neural network for multi-label classification of cardiac arrhythmias in 12-lead ECG records. Experiments on a public 12-lead ECG dataset showed the effectiveness of our method. The proposed model achieved an average area under the receiver operating characteristic curve (AUC) of 0.970 and an average F1 score of 0.813. Using single-lead ECG as model input produced lower performance than using all 12 leads. The best-performing leads are lead I, aVR, and V5 among 12 leads. Finally, we employed the SHapley Additive exPlanations (SHAP) method to interpret the model's behavior at both patient-level and population-level.

## Model Architecture

<img src="https://imgur.com/BIvuVUc.png" width="300">

> Deep neural network architecture for cardiac arrhythimas diagnosis.

## Requirement

### Dataset

The 12-lead ECG dataset used in this study is the CPSC2018 training dataset which is released by the 1st China Physiological Signal Challenge (CPSC) 2018 during the 7th International Conference on Biomedical Engineering and Biotechnology. Details of the CPSC2018 dataset can be found [here](https://bit.ly/3gus3D0). To access the processed data, click [here](https://www.dropbox.com/s/unicm8ulxt24vh8/CPSC.zip?dl=0).

### Software

- Python 3.7.4
- Matplotlib 3.1.1
- Numpy 1.17.2
- Pandas 0.25.2
- PyTorch 1.2.0
- Scikit-learn 0.21.3
- Scipy 1.3.1
- Shap 0.35.1
- Tqdm 4.36.1
- Wfdb 2.2.1

## Run

### Preprocessing

```sh
$ python preprocess.py --data-dir data/CPSC
```

### Baselines

```sh
$ python baselines.py --data-dir data/CPSC --classifier LR
```

### Deep model

```sh
$ python main.py --data-dir data/CPSC --leads all --use-gpu # training
$ python predict.py --data-dir data/CPSC --leads all --use-gpu # evaluation
```

### Interpretation

```sh
$ python shap_values.py --data-dir data/CPSC --use-gpu # visualizing shap values
```
