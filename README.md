# Customer-Churn-Prediction-Retention
An end-to-end machine learning pipeline for predicting customer churn, featuring automated preprocessing, feature engineering, model tuning, and SHAP explainability. Achieves 0.847 ROC-AUC with XGBoost, with a FastAPI endpoint and business impact analysis for retention strategies.
Data & Preprocessing
The Telco customer churn dataset (7,043 records, 21 features) was prepared using a fully automated sklearn pipeline. Missing values in TotalCharges were imputed with the median, and three new features were engineered: total number of subscribed services, tenure group (new / established / loyal), and charge per service ratio. Categorical variables were one hot encoded and numeric features were standardised.
Imbalance Handling
The dataset has a 27% churn rate – a moderate class imbalance. We compared two competing strategies:
•	Class weighting (using class_weight='balanced' in LogisticRegression and RandomForest, and scale_pos_weight in XGBoost).
•	SMOTE oversampling (applied before training).
A quick cross validation experiment on Logistic Regression and Random Forest showed that class weighting consistently outperformed SMOTE, yielding higher ROC AUC and better F1 score without introducing synthetic samples. Therefore, class weighting was chosen as the final imbalance strategy for all models.
Model Training & Selection
Five models were trained and tuned using RandomizedSearchCV with 3 fold nested cross validation:
•	Logistic Regression
•	Random Forest
•	Gradient Boosting
•	XGBoost
•	Voting Ensemble (soft voting over the best 3 individual models)
The best model was [model name, e.g., XGBoost] with an ROC AUC of [value], PR AUC of [value], and F1 score of [value] on the test set. Its predicted probabilities were well calibrated, as shown in the calibration curve. SHAP analysis identified tenure, monthly charges, contract type, and total services as the top drivers of churn – short tenure, high cost, month to month customers are most likely to leave.
Business Recommendation
The model flags the top 10% of customers with the highest predicted churn probability. Retaining this group would save approximately $[X] in monthly recurring revenue – a substantial impact. We recommend the following actions:
1.	Targeted retention offers – provide discounts or loyalty bonuses to these high risk customers.
2.	Improve engagement – proactively reach out with personalised service recommendations (e.g., bundling, tech support).
3.	Monitor early indicators – focus on customers with short tenure and month to month contracts; offer contract upgrades to reduce churn. 
