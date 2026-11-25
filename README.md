# 🌊 JALRAKSHAK AI
### Machine Learning-Based Experimental Flood Risk Prediction System

JALRAKSHAK AI is an experimental machine learning project designed to predict flood risk for **1-day, 3-day, and 7-day forecasting horizons** using river levels, reservoir levels, and rainfall-related features.

The project uses machine learning classification models to categorize flood risk into three levels:

- 🟢 GREEN – Low Risk
- 🟡 YELLOW – Moderate Risk
- 🔴 RED – High Risk

---

## 🎯 Project Objectives

- Analyze river and reservoir water-level conditions.
- Incorporate rainfall patterns into flood risk prediction.
- Predict flood risk for multiple forecasting horizons.
- Provide machine learning-based risk classification.

## 🤖 Machine Learning Models

Three trained models are included:

| Model | Forecasting Horizon | Accuracy |
|---|---|---:|
| Flood Model 1 | 1 Day | 82.14% |
| Flood Model 2 | 3 Days | 73.62% |
| Flood Model 3 | 7 Days | 71.07% |

### Model Features

The models use the following input features:

- River Level
- Reservoir Level
- River Level Change
- Reservoir Level Change
- Daily Rainfall
- 3-Day Rainfall
- 7-Day Rainfall
- Rainfall Missing Indicator

## 📁 Project Structure

```text
JALRAKSHAK-AI/
│
├── models/
│   ├── flood_final_1day.pkl
│   ├── flood_final_3day.pkl
│   └── flood_final_7day.pkl
│
├── common.py
├── flood_predict.py
├── flood_final_training.py
├── requirements.txt
└── .gitignore
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Machine Learning

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd JALRAKSHAK-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run Prediction

Execute the prediction script:

```bash
python flood_predict.py
```

Provide the required river-level, reservoir-level, and rainfall inputs when prompted.

## 📊 Model Evaluation

| Forecast | Accuracy | Macro F1-Score |
|---|---:|---:|
| 1-Day | 82.14% | 82.76% |
| 3-Day | 73.62% | 75.26% |
| 7-Day | 71.07% | 72.60% |

## ⚠️ Important Disclaimer

JALRAKSHAK AI is an experimental prototype.

The training labels are based on heuristic, percentile-based risk definitions and are not verified historical flood-event labels. Model scores are not calibrated real-world flood probabilities.

This system must not be used as an official flood warning or as a replacement for government authorities and hydrological monitoring systems.

## 👨‍💻 Project Contribution

This repository contains the Machine Learning module of the JALRAKSHAK AI project, including trained forecasting models, prediction scripts, and model training code.

---

**JALRAKSHAK AI – Towards Smarter Experimental Flood Risk Prediction.**
