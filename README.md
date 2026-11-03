# Intrusion Detection System - Final Year Project
### Produced by Ben Sheppard - 2025

## Overview
This project compares machine learning and deep learning approaches to network intrusion detection. Each script loads a public intrusion detection dataset, preprocesses it, trains several binary classifiers (normal vs. attack traffic) and reports how well they perform.

## Datasets
| Script | Dataset | Notes |
|---|---|---|
| [KDD.py](KDD.py) | NSL-KDD | Uses the separate `KDDTrain.csv` / `KDDTest.csv` files. Attack classes are collapsed to `normal` / `abnormal`. |
| [Cicids2017.py](Cicids2017.py) | CICIDS2017 | Reads `IoTNetworkIntrusionDataset.csv`. Flow ID, IPs, timestamp and category columns are dropped. |
| [TON_IoT.py](TON_IoT.py) | TON_IoT (network) | Reads `TON_IoT_Train_Test_Network.csv`. DNS, SSL, HTTP and "weird" columns are dropped; IPs, protocol, service and connection state are label encoded. |

The datasets are not included in the repository. Place the CSV files in a `Datasets/` folder in the project root (this folder is git-ignored).

## Pipeline
Each script follows the same structure:
1. **Load and clean**: read the CSV, check for nulls, handle infinite/NaN values (CICIDS2017).
2. **Exploratory data analysis**: dataset shape, head, histograms/box plots (CICIDS2017), and label distribution plots.
3. **Preprocessing**: drop irrelevant columns, label/one-hot encode categorical features, standardise with `StandardScaler`.
4. **Train/test split**: 80:20 split for CICIDS2017 and TON_IoT; for NSL-KDD the provided train set is split 80:20 for validation.
5. **Machine learning models**: Random Forest (50 trees), Decision Tree, Gradient Boosting and XGBoost.
6. **Deep learning model**: a Keras sequential DNN (Dense 256 → BatchNorm → Dense 128 → Dense 64 → sigmoid output), trained for 10 epochs with Adam and binary cross-entropy.
7. **Evaluation**: accuracy, precision, recall, F1, confusion matrix and training time per model, followed by a seaborn bar chart comparing all models.

## Requirements
- Python 3
- pandas, numpy
- scikit-learn
- xgboost
- tensorflow
- matplotlib, seaborn

Install with:
```bash
python -m venv fyp_env
source fyp_env/bin/activate
pip install -r requirements.txt
```

## Usage
Add the dataset files to `Datasets/`, then run the script for the dataset you want:
```bash
python KDD.py
python Cicids2017.py
python TON_IoT.py
```
Plots are shown in interactive windows; close each one to continue. Metrics are printed to the console.
