# Loan Approval Prediction Using Logistic Regression

This project uses a supervised machine learning approach to predict whether a loan application will be approved based on customer details such as income, employment status, credit history, and more.

## Dataset

The dataset includes the following features:
- Gender
- Marital Status
- Dependents
- Education
- Self Employed
- Applicant Income
- Co-applicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area
- Loan Status (target variable)

---

## Data Preprocessing

- Handled missing values using:
  - **Mode** for categorical features like `Gender`, `Married`, etc.
  - **Median** for `LoanAmount`
- Encoded categorical variables using **one-hot encoding** (`pd.get_dummies`)
- Dropped irrelevant columns like `Loan_ID`
- Scaled numeric features using `StandardScaler` for better model performance

---

## Exploratory Data Analysis

- Checked for class imbalance in the target variable (`Loan_Status`)
- Used heatmap to identify correlation between numerical features
- Observed strong correlation between `Credit_History` and loan approval

---

## Model Training

- Used **Logistic Regression** from `sklearn`
- Performed a **70-30 train-test split**
- Scaled features to help with convergence
- Increased `max_iter` to ensure convergence if necessary

```python
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
```
---

## Results

Achieved 100% accuracy on test set

Confusion Matrix:
            [[ 65   0]
             [  0 120]]

---

## Considerations

High accuracy might indicate:
- A simple, linearly separable dataset Or potential data leakage

Data pipeline was double-checked to avoid leakage

Future improvements can include:
- Cross-validation
- Trying advanced models like Random Forest or XGBoost
- Model deployment (Flask, Streamlit)
  
