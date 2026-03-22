# Capstone Project — Loan Default Risk Analysis

## Executive Summary

The goal of this project is to build a model that can identify loans that are likely to default. This is an important problem because missed defaults can lead to direct financial losses for lenders.

Several machine learning models were tested using Lending Club loan data. The main focus was on recall, since correctly identifying risky borrowers is more important than minimizing false alarms.

After evaluating multiple models, Logistic Regression turned out to be the most effective for this objective. It was able to identify a large portion of default cases, while more complex models such as Random Forest and LightGBM failed to detect defaults reliably.

An important takeaway from this project is that simpler models can sometimes outperform more complex ones when they are aligned with the right business objective.

---

## Business Problem

Loan defaults are costly, and lenders need a way to identify high-risk borrowers before issuing credit.

The main question behind this project is:

How can we improve the detection of risky loans before losses occur?

Because the cost of missing a default is high, the model is evaluated primarily based on recall.

---

## Dataset

The dataset used in this project comes from Lending Club and is available on Kaggle.

File used:
- loan.csv

It includes information about:
- loan characteristics (amount, interest rate, term)
- borrower profile (income, employment length)
- credit behavior (debt-to-income ratio, delinquencies, credit history)
- account activity (balances, utilization, number of accounts)

---

## Repository Contents

- README.md  
- LoanDefaultModeling.ipynb  
- LoanDefaultEvaluation.ipynb  

---

## Approach

The project follows a standard machine learning workflow.

### Data Preparation
- Removed irrelevant columns and potential data leakage variables
- Converted percentages and text fields into numeric format
- Handled missing values
- Filtered out unrealistic values

### Exploratory Analysis
- Reviewed distributions of key variables
- Compared default vs non-default behavior
- Identified patterns in risk drivers such as interest rate, debt burden, and credit history

### Feature Engineering
Created additional features to better represent borrower risk, including:
- loan-to-income ratio
- payment-to-income ratio
- credit history length
- delinquency indicators
- utilization flags

### Modeling
Tested multiple models:
- Logistic Regression
- Random Forest
- LightGBM

### Validation
- Train/test split with stratification
- Cross-validation on training data
- Hyperparameter tuning using RandomizedSearchCV and GridSearchCV

### Evaluation
Models were evaluated using:
- recall (primary metric)
- precision
- F1-score
- ROC-AUC
- confusion matrices
- threshold analysis

---

## Model Performance (Test Set)

| Model                | Recall | Precision | F1-score | ROC-AUC |
|---------------------|--------|-----------|----------|---------|
| Logistic Regression | 0.619  | 0.164     | 0.260    | 0.675   |
| Random Forest       | 0.033  | 0.189     | 0.057    | 0.640   |
| LightGBM            | 0.000  | 0.000     | 0.000    | 0.677   |

### Key Observations

Logistic Regression performed best in terms of recall, identifying about 62% of default cases.

Random Forest and LightGBM achieved higher accuracy but failed to detect defaults effectively. In particular, LightGBM classified almost all loans as non-default, resulting in zero recall.

This highlights an important point: accuracy alone is not a reliable metric for imbalanced problems like loan default prediction.

---

## Threshold Tuning

The model outputs probabilities, and a threshold is used to decide whether a loan is classified as risky.

Lowering the threshold increases recall but also increases false positives. Raising the threshold reduces false positives but increases the chance of missing defaults.

For Logistic Regression:

- At threshold 0.5: recall is about 0.62  
- At threshold 0.3: recall increases to about 0.94  
- At threshold 0.2: recall reaches about 0.99  

This means the model can be adjusted depending on how conservative the lender wants to be.

---

## Final Recommendation

Logistic Regression is the recommended model for this use case.

It provides the best balance for a recall-focused problem and performs consistently across validation and test data.

From a business perspective, it is better to flag more loans for review than to miss high-risk borrowers. The model can be used as a support tool for:

- underwriting review prioritization  
- risk-based pricing  
- manual review escalation  
- portfolio risk monitoring  

The threshold can be adjusted depending on the organization’s risk tolerance and operational capacity.

---

## Key Insight

More complex models do not always perform better.

In this case, Logistic Regression outperformed tree-based models in identifying defaults. This suggests that the main drivers of default risk in this dataset are relatively straightforward and can be captured by a linear model.

---

## Limitations

- Current loans were treated as non-default, which may not always be accurate
- No time-based validation was performed
- No external or macroeconomic variables were included
- The model does not explicitly incorporate business cost (only recall as a proxy)

---

## Next Steps

- Add time-based validation
- Explore cost-sensitive modeling
- Improve probability calibration
- Test additional features or external data
- Use explainability techniques such as SHAP

---



