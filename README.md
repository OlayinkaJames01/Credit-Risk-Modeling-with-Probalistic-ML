# 🧠 Retail Credit Risk Modeling: Predicting Customer Defaults for Kowope Mart

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

This project applies machine learning to predict customer credit defaults for **Kowope Mart**, a Nigerian retail company offering co-branded credit cards in partnership with DSBank. The goal is to identify high-risk customers and reduce default rates without limiting credit access.

## 📊 Dataset Summary

- **Source:** Zindi Africa - Data Science Nigeria AI Bootcamp Qualification Hackathon  
- **Train Samples:** 56,000  
- **Test Samples:** 24,000  
- **Target Variable:** `default_status` (1 = Defaulted, 0 = Non-Default)  
- **Features:** 50+ structured variables on credit usage, risk, loan severity, and time-based activity

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `seaborn`  
- **ML Tasks:**  
  - Data preprocessing & cleaning  
  - Feature engineering  
  - Model training & evaluation (Logistic Regression, XGBoost, LightGBM)
  - Hyperparameter tunning (Stratified K-Fold) 
  - Probabilistic classification (Switching from binary classification to probablity of customer defaulting for real life application)

## ✅ Outcomes

The model provides a robust tool for classifying customers likely to default, improving credit policy decisions and risk profiling.

---
**Challenge Link:** [Zindi - DSN AI Bootcamp Hackathon](https://zindi.africa/hackathons/dsn-ai-bootcamp-qualification-hackathon)  
