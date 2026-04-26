# Deep Learning-Based Intrusion Detection System (CNN-LSTM)

## 📌 Overview
This project presents a Deep Learning-based Intrusion Detection System (IDS) designed to identify malicious network activity with high accuracy. The system leverages a hybrid architecture combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to detect complex and evolving cyber threats in network traffic data.

The goal is to address limitations of traditional rule-based and signature-based IDS, particularly in detecting zero-day attacks and reducing false positive rates.

---

## 🚨 Problem Statement
Traditional Intrusion Detection Systems struggle with:
- High false positive rates
- Poor detection of evolving cyber threats
- Inability to learn from large-scale, heterogeneous network data

This project aims to overcome these challenges using deep learning techniques that automatically extract meaningful patterns from raw network traffic.

---

## 💡 Proposed Solution
A hybrid deep learning architecture is implemented:
- **CNN Layers**: Extract spatial features from network traffic data
- **LSTM Layers**: Capture temporal dependencies and sequential attack patterns
- **Fully Connected Layers**: Perform classification of normal vs malicious traffic

This combination allows the system to learn both **feature-level patterns** and **time-based attack behaviors**.

---

## 🧠 Key Features
- Automated feature extraction (no manual feature engineering required)
- Scalable architecture for large datasets
- Adaptability to new and evolving attack patterns
- Reduced false positives through deep pattern recognition
- End-to-end pipeline: preprocessing → training → evaluation

---

## 📊 Dataset
The model is trained and evaluated using the:
- NSL-KDD dataset (benchmark dataset for intrusion detection)

> Note: The dataset includes labeled network traffic with multiple attack categories such as DoS, Probe, R2L, and U2R.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Removal of missing values
- Feature normalization using StandardScaler
- Reshaping data for CNN-LSTM input

### 2. Model Architecture
- Conv1D + MaxPooling layers for feature extraction
- LSTM layer for sequence learning
- Dense layers for classification

### 3. Training
- Binary classification (normal vs attack)
- Loss Function: Binary Crossentropy
- Optimizer: Adam

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## 📈 Results
The model demonstrates strong performance in intrusion detection tasks:

- High classification accuracy
- Improved detection of malicious traffic
- Reduced false positive rate compared to traditional methods

Example outputs include:
- Confusion Matrix
- ROC Curve
- Accuracy & Loss plots
