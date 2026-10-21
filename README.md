
# 🫀 Atrial Fibrillation Detection using 1D CNN-BiLSTM

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework for automated **Atrial Fibrillation (AFib)** detection from single-lead Electrocardiogram (ECG) signals using a hybrid **1D CNN-BiLSTM** neural network.

This project combines records from the **MIT-BIH Atrial Fibrillation Database** and the **PTB-XL ECG Database**, standardizes single-lead ECG signals to 360 Hz, addresses class imbalance using SMOTE oversampling, and evaluates clinical metrics including Sensitivity (Recall), Precision, and AUC-ROC.

---

## 📊 Key Performance Metrics

The primary **1D CNN-BiLSTM** model achieved strong performance on the held-out test set:

| Metric | Score | Clinical Relevance |
| :--- | :---: | :--- |
| **Test Accuracy** | **90.30%** | Overall correct classification rate |
| **Recall (Sensitivity)** | **93.06%** | High sensitivity minimizes missed AFib events |
| **Precision** | **85.46%** | High precision limits false positive alarms |
| **AUC-ROC** | **0.9199** | Strong discrimination across classification thresholds |

---

## 🏗️ Repository Architecture

```text
atrial-fibrillation-detection/
├── .gitignore                  # Excludes raw data and model binaries
├── README.md                   # Project documentation
├── requirements.txt            # Dependency specification
│
├── data/                       # Dataset storage (local/ignored)
│   ├── raw/                    # MIT-BIH & PTB-XL raw files
│   └── processed/              # Processed signal arrays
│
├── src/                        # Modular Python source package
│   ├── data/
│   │   ├── mitbih_loader.py    # WFDB signal extraction & 10s segmentation
│   │   ├── ptbxl_loader.py     # PTB-XL Lead II extraction & scp_codes parsing
│   │   └── preprocessing.py    # Unification, StandardScaler, & SMOTE
│   │
│   ├── models/
│   │   ├── cnn_bilstm.py       # Primary 1D CNN-BiLSTM Keras model
│   │   └── cnn_lstm.py         # Baseline 1D CNN-LSTM model
│   │
│   ├── training/
│   │   └── train.py            # End-to-end training pipeline with callbacks
│   │
│   └── utils/
│       └── visualization.py    # Confusion matrix, ROC, PR curves, confidence plots
│
├── saved_models/               # Trained weights (.keras) & fitted scaler (.pkl)
└── outputs/                    # Exported evaluation figures & diagnostic plots
```
