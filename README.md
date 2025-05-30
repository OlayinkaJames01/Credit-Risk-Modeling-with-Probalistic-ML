# 🧠 Retail Credit Risk Modeling: Predicting Customer Defaults for Kowope Mart

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

This project applies machine learning to predict customer credit defaults for **Kowope Mart**, a Nigerian retail company offering co-branded credit cards in partnership with DSBank. The goal is to identify high-risk customers and reduce default rates without limiting credit access.

## 📊 Dataset Summary

- **Source:** Zindi Africa - DSN AI Bootcamp Qualification Hackathon  
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
  - Model training & evaluation (XGBoost, LightGBM)  
  - Binary classification metrics: AUC, Accuracy, Confusion Matrix

## ✅ Outcomes

The model provides a robust tool for classifying customers likely to default, improving credit policy decisions and risk profiling.




###### Hackathon Link
https://zindi.africa/hackathons/dsn-ai-bootcamp-qualification-hackathon

Kowope Mart is a Nigerian-based retail company with a vision to provide quality goods, education and automobile services to its customers at affordable price and reduce if not eradicate charges on card payments and increase customer satisfaction with credit rewards that can be used within the Mall. To achieve this, the company has partnered with DSBank on co-branded credit card with additional functionality such that customers can request for loan, pay for goods even with zero-balance and then pay back within an agreed period of time. This innovative strategy has increased sales for the company. However, there has been recent cases of credit defaults and Kowope Mart will like to have a system that profiles customers who are worthy of the card with minimum if not zero risk of defaulting. 

You have been employed as a Data Scientist to leverage Machine learning to predict customers who are likely to default or not.

#### Dataset Information
The data is a structure dataset pull from the Kowope Mart’s database with few important features that can be used to predict defaulters in retail network. The data sample has been split into train and test with a volume of 56,000 and 24,000.

##### The files for download are:
train.csv - you will use this to train your model.
test.csv - you will use this to test your model.
SampleSubmission.csv - is an example of what your submission file should look like. The order of the rows does not matter, but the names of the UIDs must be correct. You must predict growth_stage for the corresponding images.

#### Variables
Application_ID: Unique Customer Application Identification number

#### Risk Scoring features:
form_field1: Customer Creditworthiness score based on historical data
form_field2: A score that measures the number and riskiness of credit enquiries made by a borrower.

#### Severity features: Severity is a function of amount, time since default, and number of defaults
form_field3: Severity of default by the borrower on any loan(s).
form_field4: Severity of default by the borrower on auto loan(s).
form_field5: Severity of default by the borrower on education loan(s).

#### Credit features
form_field6: Minimum of credit available on all credit cards that is automatically renewed as debts are paid off on the customer’s cards (in NGN)
form_field7: Maximum of credit available on customer’s active credit lines (in NGN)
form_field8: Maximum of credit available on all active credit cards that is automatically renewed as debts are paid off on the customer’s cards (in NGN)
form_field9: Sum of available credit on credit cards that the borrower has missed 1 payment (in NGN)
form_field10: Total amount of credit available on accepted credit lines (in NGN)
form_field11: The amount of dues collected post-default where the due amount was more than 500 (in NGN)
form_field12: Sum of the amount due on active credit cards (in NGN)
form_field13: Annual amount paid towards all credit cards during the previous year (in NGN)
form_field14: Annual income (in NGN)
form_field15: The estimated market value of a property owned/used by the borrower (in NGN)

#### Credit card features:
form_field16: Number of active credit card that is automatically renewed as debts are paid off on which full credit limit is utilized by the borrower
form_field17: Number of active credit cards on which full credit limit is utilized by the borrower
form_field18: Number of active credit lines on which full credit limit is utilized by the borrower
form_field19: Number of active credit cards on which at least 75% credit limit is utilized by the borrower
form_field20: Number of active credit lines on which at least 75% credit limit is utilized by the borrower
form_field21: Average utilization of active revolving credit card loans (%)
form_field22: Average utilization of line on all active credit lines activated in last 2 years (%)
form_field23: Average utilization of line on all active credit cards activated in last 1 year (%)
form_field24: Average utilization of line on credit cards on which the borrower has missed 1 payment during the last 6 months (%)
form_field25: Average tenure of active revolving credit cards (in days)
form_field26: Tenure of oldest credit card among all active credit cards (in days)
form_field27: Tenure of oldest revolving credit card among all active revolving credit cards (in days)

#### Time-related features
form_field28: Number of days since last missed payment on any credit line
form_field29: Tenure of the oldest credit line (in days)
form_field30: Maximum tenure on all auto loans (in days)
form_field31: Maximum tenure on all education loans (in days)
form_field32: Sum of tenures (in months) of active credit cards
form_field33: Sum of tenures (in months) of active credit cards
form_field34: Number of active credit lines over the last 6 months on which the borrower has missed 1 payment
form_field35: Number of revolving credit cards over the last 2 years on which the borrower has missed 1 payment
form_field36: Number of active credit lines
form_field37: Number of credit cards with an active tenure of at least 2 years
form_field38: Number of credit lines activated in the last 2 years
form_field39: Number of credit lines on which the borrower has current delinquency

#### Percentage related features
form_field40: Utilization of line on active education loans (%)
form_field41: Utilization of line on active auto loans (%)

#### Other features
form_field42: Financial stress index of the borrower. This index is a function of collection trades, bankruptcies files, tax liens invoked, etc.
form_field43: Number of credit lines on which the borrower has never missed a payment in the last 2 years, yet considered as high-risk loans based on the market prediction of the economic scenario
form_field44: Ratio of the maximum amount due on all active credit lines and the sum of amounts due on all active credit lines
form_field45: Number of mortgage loans on which the borrower has missed 2 payments
form_field46: Number of auto loans on which the borrower has missed 2 payments
form_field47: Type of product that the applicant applied for. (C = Charge; L = Lending)
form_field48: Undefined Variable
form_field49: Undefined Variable
form_field50: Ratio of the minimum amount due on all active credit lines and the sum of amounts due on all active credit lines

#### Target features
default_status: defaulted or not. (yes:1, no: 0)

