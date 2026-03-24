# 🫀 Brugada Syndrome Detection from 12-Lead ECG
### IDSC 2026 — *Mathematics for Hope in Healthcare* | Team **Emebege**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Competition](https://img.shields.io/badge/IDSC-2026-purple?style=flat-square)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebook/brugada_detection.ipynb)

> Automated detection of Brugada Syndrome using deep learning on 12-lead ECG signals, combining CNN-Attention and BiLSTM architectures with explainability via SHAP and Grad-CAM.

---

## 👥 Team

| Name | Role |
|------|------|
| Fherenilla Anandianus | Model Development |
| Reiviena Bellia Agitha | Feature Engineering |
| Ismatul Izza | Data Analysis |

---

## 📋 Overview

Brugada Syndrome is a rare but life-threatening cardiac arrhythmia disorder, often missed in routine ECG interpretation. This project builds an end-to-end deep learning pipeline to **automatically detect Brugada Syndrome** from 12-lead ECG signals — enabling faster, more consistent clinical screening.

Our approach covers the full machine learning lifecycle:
- Signal preprocessing and quality evaluation
- Statistical feature extraction and selection
- Deep learning modeling with class imbalance handling
- Explainability with SHAP and Grad-CAM

---

## 📂 Dataset

- **Source:** [Brugada-HUCA Dataset](https://physionet.org/) — PhysioNet
- **Signals:** 12-lead ECG, 100 Hz sampling rate
- **Labels:** `1` = Brugada Syndrome, `0` = Normal
- **Format:** WFDB (`.hea` + `.dat`) + `metadata.csv`

> ⚠️ Dataset is not included in this repository due to file size. Please download from PhysioNet and place it in your Google Drive at `MyDrive/embege/brugada-huca/`.

---

## 🧠 Methods

![Pipeline](pipeline_emebege_idsc2026.svg)

### 1. Preprocessing
- Bandpass filter: **0.5–40 Hz** (Butterworth, order 4)
- Removal of anomalous records
- Signal quality evaluation (SNR, amplitude range)

### 2. Feature Extraction
Time-domain and frequency-domain features extracted per lead:
- RMS, STD, amplitude range
- ST segment min/std
- PSD low-frequency to high-frequency ratio (`psd_lf_hf`)
- Statistical significance tested via **Mann-Whitney U** + FDR correction

### 3. Modeling
Two architectures trained **with and without SMOTE** oversampling:

| Model | Description |
|-------|-------------|
| **CNN-Attention** | 1D Convolutional + Channel Attention mechanism |
| **BiLSTM** | Bidirectional LSTM for temporal pattern capture |

- Loss: **Focal Loss** (γ=2.0, α=0.25) to handle class imbalance
- Threshold optimization with minimum sensitivity constraint ≥ 0.80

### 4. Explainability
- **SHAP** — feature importance for clinical interpretability
- **Grad-CAM (1D)** — highlights ECG timesteps the model focuses on

---

## 📊 Results

> Evaluated at optimal decision threshold (sensitivity ≥ 0.80 constraint).

| Model | Threshold | AUC | F1 | Sensitivity | Specificity | PPV | NPV |
|-------|-----------|-----|----|-------------|-------------|-----|-----|
| CNN-Attention (no SMOTE) | 0.27 | 0.922 | 0.800 | 0.909 | 0.907 | 0.714 | 0.975 |
| BiLSTM (no SMOTE) | 0.26 | 0.922 | 0.667 | **1.000** | 0.744 | 0.500 | 1.000 |
| **CNN-Attention (SMOTE)** ⭐ | **0.10** | **0.924** | **0.833** | **0.909** | **0.930** | **0.769** | **0.976** |
| BiLSTM (SMOTE) | 0.23 | 0.882 | 0.643 | 0.818 | 0.814 | 0.529 | 0.946 |

**Best model: CNN-Attention (SMOTE)** — highest AUC (0.924), F1 (0.833), and Specificity (0.930) with 91% sensitivity.



---

## 🗂️ Repository Structure

```
.
├── notebook/
│   └── brugada_detection.ipynb   # Main notebook (full pipeline)
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 🚀 How to Run

### ▶️ Option 1 — Google Colab (Recommended)

Click the badge below to open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebook/brugada_detection.ipynb)

Then follow these steps inside the notebook:

**Step 1 — Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Step 2 — Prepare the dataset**

Place the downloaded dataset in your Google Drive at:
```
MyDrive/
└── embege/
    └── brugada-huca/
        ├── files/           # WFDB signal files (one folder per patient)
        └── metadata.csv     # Patient labels and clinical metadata
```

**Step 3 — Run all cells**

Go to **Runtime → Run All** and the full pipeline will execute automatically.

---

### 💻 Option 2 — Local Environment

```bash
git clone https://github.com/fherenillaa/Detection-of-Brugada-Syndrome-from-12-Lead-ECG.git
cd Detection-of-Brugada-Syndrome-from-12-Lead-ECG
pip install -r requirements.txt
jupyter notebook notebook/brugada_detection.ipynb
```

> Note: Local setup requires adjusting dataset paths inside the notebook.

---

## 📦 Requirements

```
wfdb
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
imbalanced-learn
tensorflow>=2.10
shap
tqdm
statsmodels
```

---

## 🔬 Clinical Relevance

Brugada Syndrome affects an estimated **1 in 2,000** people globally and is a leading cause of sudden cardiac death in young adults. Automated ECG screening tools can:
- Reduce missed diagnoses in underserved clinical settings
- Provide decision support for non-specialist clinicians
- Enable large-scale population screening programs

---

## 📄 License

This project is submitted as part of **IDSC 2026** — International Data Science Challenge.  
Code is released under the [MIT License](LICENSE).

---

## 🔗 References

