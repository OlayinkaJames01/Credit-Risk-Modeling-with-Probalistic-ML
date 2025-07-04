import logging
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # Load cleaned data
    cleaned_train_df = pd.read_pickle('../data/cleaned_train.pkl')
    cleaned_test_df = pd.read_pickle('../data/cleaned_test.pkl')

    # Prepare features and target
    X = cleaned_train_df.drop(columns=['default_status', 'Applicant_ID'])
    y = cleaned_train_df['default_status']
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    logger.info("Logistic Regression model trained.")

    # Predict probabilities on validation set
    y_val_probs = model.predict_proba(X_val_scaled)[:, 1]

    # Evaluate ROC-AUC
    roc_auc = roc_auc_score(y_val, y_val_probs)
    logger.info(f"Validation ROC-AUC score: {roc_auc:.4f}")

    # Predict classes at 0.5 threshold
    y_val_preds = (y_val_probs >= 0.5).astype(int)

    # Log classification report and confusion matrix
    logger.info("Classification report:\n" + classification_report(y_val, y_val_preds))
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_val, y_val_preds)}")

    # Prepare test data and predict
    X_test = cleaned_test_df.drop(columns=['Applicant_ID'])
    X_test_scaled = scaler.transform(X_test)
    test_probs = model.predict_proba(X_test_scaled)[:, 1]

    # Save predictions
    result = pd.DataFrame({
        'Applicant_ID': cleaned_test_df['Applicant_ID'],
        'default_status': test_probs
    })
    result.to_csv('../outputs/logistic_regression.csv', index=False)
    logger.info("Test set predictions saved to ../outputs/logistic_regression.csv")

if __name__ == '__main__':
    main()
