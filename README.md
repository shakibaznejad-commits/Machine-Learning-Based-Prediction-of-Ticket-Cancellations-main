# Machine Learning-Based Prediction of Ticket Cancellations for MrBilit

Project Creator: Shakiba Zakeri Nejad

## Overview

This project aims to predict user ticket cancellations for MrBilit, helping the company mitigate penalty fees charged by transportation providers. An XGBoost classification model was developed and trained on historical trip data.

## Key Results

* Accuracy (Test Set): 98%
* F1-Score (Cancellation Class): 0.94

## Methodology

The project involved the following key steps:
1.  Data Loading & Preparation: Reading the dataset, handling date/time features, and splitting into training/testing sets.
2.  Preprocessing: Managing missing values (including imputation for VehicleClass using a secondary XGBoost model), cleaning invalid data (e.g., non-positive prices), and encoding categorical features (TripReason, Male, CouponDiscount, From, To, Vehicle).
3.  Feature Engineering: Creating the TimeDifference_minutes feature representing the duration between booking and departure.
4.  Scaling: Applying StandardScaler to numerical features.
5.  Modeling: Training an XGBClassifier.
6.  Hyperparameter Tuning: Using GridSearchCV to optimize n_estimators, learning_rate, and max_depth based on F1-score. (Optimal: {'learning\_rate': 0.1, 'max\_depth': 8, 'n\_estimators': 300})
7.  Evaluation: Assessing performance on the test set using accuracy, F1-score, classification report, and confusion matrix.

## Dataset

The dataset contains anonymized information about tickets, users, and trips from MrBilit, sourced from Kaggle (mrbilit_dataset.csv).

## Key Features Used

* ReserveStatus
* Male
* Price
* CouponDiscount (Categorized & Encoded)
* Domestic
* Vehicle (Encoded)
* VehicleClass (Imputed & Encoded)
* From (Encoded)
* To (Encoded)
* TimeDifference_minutes (Engineered)
* TripReason (Encoded)

The target variable is Cancel (1 for cancelled, 0 otherwise). Feature importance analysis indicated VehicleClass and ReserveStatus as the most influential predictors.

## Installation

Clone the repository and install the required dependencies using the provided requirements.txt file:

```bash
git clone [https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories)
cd [repository directory]
pip install -r requirements.txt


Results
The model achieved high accuracy (98%) and a strong F1-score (0.94) for predicting cancellations on the unseen test data. Detailed performance metrics, including the confusion matrix and classification report, can be found within the notebook.
