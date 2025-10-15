# 🏥 Hospital Readmission Prediction System

**Predicting 30-Day Hospital Readmissions using Medical + Social Determinants of Health**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Dashboard](https://img.shields.io/badge/Dashboard-Dash-blue.svg)

---

A comprehensive machine learning project that predicts **30-day hospital readmissions** by integrating **clinical data** with **Social Determinants of Health (SDOH)**. The system aims to address healthcare disparities and enhance prediction accuracy using real-world health and community-level indicators.

---

## 🎯 Key Features

- 🤖 **Multi-Model Approach**  
  Logistic Regression, Random Forest, XGBoost with hyperparameter tuning

- 📊 **Comprehensive Data Integration**  
  Medical records + CDC Social Vulnerability Index (SVI)

- ⚖️ **Class Imbalance Handling**  
  SMOTE for balanced training data

- 🎮 **Interactive Dashboard (Dash)**  
  Real-time patient risk prediction with Plotly Dash

- 🔍 **Model Interpretability**  
  SHAP values, feature importance, and model explanations

- 🧭 **Fairness & Bias Analysis**  
  Equity evaluation across demographic subgroups

- 🏥 **Real-Time Prediction Utility**  
  Instant risk assessment to support care planning

---

## 🚀 Quick Start

### ✅ Prerequisites

- Python **3.8+**
- `pip` package manager

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/PallySlakk/predictor-project.git
cd predictor-project

▶️ Run the Complete System
python3 main.py

This single command will:

✅ Install required dependencies

📥 Load & preprocess integrated datasets

🤖 Train predictive ML models

📊 Launch the interactive dashboard at http://localhost:8050

🎮 Dashboard Highlights

Use the dashboard to:

🩺 Assess Patient Risk – Input clinical & social factors for real-time predictions

🧪 Compare ML Models – View performance metrics across algorithms

🧭 Interpret Features – SHAP charts & feature importance

🏷 Actionable Interventions – Risk-based recommendations

📈 Results & Impact
Key Insight	Outcome
SDOH Integration	+12% improvement over clinical-only models
Top Predictors	Social Vulnerability Index in top 5 features
Model Fairness	Performance preserved across demographics
Usability	Real-time clinical decision support via dashboard

📄 License

This project is licensed under the MIT License.

⭐ Acknowledgements

CDC Social Vulnerability Index (SVI)

Healthcare providers & research datasets

Open-source ML and visualization communities
