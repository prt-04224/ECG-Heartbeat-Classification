# ECG Heartbeat Classification Using 1D ResNet and Grad-CAM

A deep learning-based ECG heartbeat classification system that combines a one-dimensional Residual Convolutional Neural Network (1D ResNet) with RR-interval features. The project also incorporates Grad-CAM for visual interpretation of model predictions and provides an interactive interface developed using Streamlit.

## 1. Project Overview

Electrocardiogram (ECG) signals provide important information about the electrical activity of the heart. This project focuses on automated heartbeat classification using deep learning and biomedical signal processing techniques.

The system processes ECG recordings, extracts individual heartbeat segments, incorporates RR-interval timing information, and classifies heartbeats into four categories.

The model is trained and evaluated using an inter-patient data split on the MIT-BIH Arrhythmia Database to assess its performance on previously unseen patients.

## 2. Classification Categories

| Class | Description |
|:---:|---|
| N | Normal-type beat |
| S | Supraventricular ectopic beat |
| V | Ventricular ectopic beat |
| F | Fusion beat |

## 3. Methodology

The proposed classification architecture consists of two feature-processing branches.

**ECG Waveform Branch**
- One-dimensional convolutional neural network.
- Residual blocks for hierarchical feature extraction.
- Global average pooling and global max pooling.

**RR-Interval Feature Branch**
- Processes four RR-related timing features.
- Uses fully connected layers to learn temporal representations.

**Classification Head**

The extracted waveform and RR-feature representations are concatenated and passed through fully connected layers to predict one of the four heartbeat categories.

### Model Specifications

| Parameter | Value |
|---|---|
| Dataset | MIT-BIH Arrhythmia Database |
| Sampling frequency | 360 Hz |
| Heartbeat segment length | 216 samples |
| RR features | 4 |
| Architecture | 1D ResNet with RR-feature branch |
| Output classes | 4 |

## 4. Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is incorporated to visualize waveform regions contributing to the model's selected class score.

This provides an interpretation of the learned representations in the waveform branch. The visualization does not represent the contribution of the separate RR-feature branch.

## 5. Application Features

The Streamlit application provides the following functionality:

- **Demo Mode:** Explore annotated heartbeat samples from unseen test patients.
- **ECG Recording Upload:** Analyse single-lead ECG recordings in CSV format.
- **R-Peak Detection:** Supports WFDB XQRS and a built-in detection method.
- **Heartbeat Classification:** Predicts the heartbeat category for individual segments.
- **RR-Interval Analysis:** Incorporates heartbeat timing features into classification.
- **Grad-CAM Visualization:** Displays waveform regions influencing model predictions.
- **Interactive Results:** Presents class scores, heartbeat distributions, recording statistics and ECG visualizations.

## 6. Model Evaluation

The model is evaluated using an inter-patient test split.

The evaluation considers:

- Accuracy
- Macro F1-score
- Class-wise precision
- Class-wise recall
- Class-wise F1-score
- Confusion matrix

Performance differs across the four classes. In particular, the Fusion (F) category has substantially weaker classification performance due to its limited representation in the training data.

Overall accuracy should therefore be considered alongside class-wise evaluation metrics.

## 7. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Numerical Computing | NumPy |
| Data Processing | Pandas, SciPy |
| ECG Processing | WFDB |
| Visualization | Matplotlib |
| Explainable AI | Grad-CAM |
| Web Framework | Streamlit |
| Version Control | Git, GitHub |

## 8. Repository Structure

```text
ECG-Heartbeat-Classification/
│
├── app.py
├── ecg_pipeline.py
├── requirements.txt
├── README.md
│
└── deploy/
    ├── config.json
    ├── demo_samples.npz
    ├── ecgnet_weights.pth
    └── preprocessing.npz
```

## 9. Installation and Execution

**Clone the repository**

```bash
git clone https://github.com/prt-04224/ECG-Heartbeat-Classification.git
cd ECG-Heartbeat-Classification
```

**Create a virtual environment**

```bash
python -m venv .venv
```

**Activate the environment (Windows)**

```bash
.venv\Scripts\activate
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run the application**

```bash
streamlit run app.py
```

The application will be available locally at `http://localhost:8501`.

## 10. Dataset

**MIT-BIH Arrhythmia Database**

PhysioNet: https://physionet.org/content/mitdb/1.0.0/

Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals", *Circulation*, 2000.

## 11. Limitations

- The system is intended for academic and research purposes only.
- It is not a medical device and must not be used for clinical diagnosis.
- Performance may vary across ECG leads, devices, sampling rates and signal quality.
- Automatic R-peak detection may introduce errors during recording analysis.
- Fusion (F) heartbeat classification remains unreliable due to limited training representation.
- Model scores should not be interpreted as clinical probabilities.

## 12. Author

**Pradyuman Manoj Tripathi**  
B.Tech, Computer Science and Engineering  
Dr. B. R. Ambedkar National Institute of Technology, Jalandhar

---

*Academic project focused on biomedical signal processing, deep learning and explainable artificial intelligence.*
