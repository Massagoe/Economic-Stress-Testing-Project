# Economic Stress Testing & Credit Risk Intelligence Platform

## Overview

This project explores how economic conditions affect loan default risk. Using more than 1.3 million Lending Club loan records and economic indicators from the Federal Reserve (FRED), I built predictive models to identify borrowers most likely to default and evaluated how portfolio risk changes under different economic scenarios.

The goal was not only to predict defaults, but also to understand how borrower characteristics and broader economic conditions influence credit risk.

---

## Business Problem

Lenders face significant losses when borrowers fail to repay their loans, especially during periods of economic uncertainty. Traditional credit models often focus on borrower information alone and may not fully account for changes in unemployment, interest rates, or inflation.

This project investigates whether combining borrower data with macroeconomic indicators can improve risk assessment and support better lending decisions.

---

## Dataset

The project combines two data sources:

### Lending Club Loan Data

Approximately 1.35 million loan records containing:

- Loan amount
- Income (revenue)
- Debt-to-income ratio (DTI)
- FICO score
- Employment experience
- Loan purpose
- Home ownership status
- Default outcome

### Federal Reserve Economic Data (FRED)

Three economic indicators were incorporated:

- **UNRATE** – U.S. Unemployment Rate
- **FEDFUNDS** – Federal Funds Rate
- **CPIAUCSL** – Consumer Price Index (Inflation)

---

## Project Workflow

### 1. Data Preparation

- Cleaned loan and economic datasets
- Converted and aligned dates
- Merged monthly economic indicators with loan records
- Removed unnecessary variables and handled missing values

### 2. Exploratory Data Analysis

Explored:

- Default distribution
- FICO score patterns
- Debt-to-income ratios
- Economic trends over time
- Relationships between risk factors and default behavior

### 3. Feature Engineering

Created several additional features, including:

- Loan-to-Income Ratio
- DTI Risk Buckets
- FICO Categories
- Economic Stress Index

The Economic Stress Index combines unemployment, inflation, and interest rate information into a single measure of economic pressure.

### 4. Predictive Modeling

Two machine learning models were developed:

- Logistic Regression
- Random Forest

Both models were evaluated using ROC-AUC, accuracy, precision, recall, and confusion matrices.

### 5. Economic Stress Testing

The models were tested under simulated economic conditions to examine how portfolio risk might change during periods of economic stress.

---

## Results

### Model Performance

| Model | ROC-AUC |
|---------|---------|
| Logistic Regression | 0.648 |
| Random Forest | 0.662 |

The Random Forest model slightly outperformed Logistic Regression, although the improvement was relatively small.

---

## Key Findings

Several important insights emerged from the analysis:

- **FICO score** was the strongest predictor of loan performance.
- Borrowers with higher incomes were less likely to default.
- Larger loan amounts were associated with higher default risk.
- Higher debt-to-income ratios significantly increased the likelihood of default.
- Macroeconomic indicators contributed useful context, but borrower-level financial characteristics had a greater influence on prediction performance.

---

## Economic Stress Testing

Stress-testing scenarios were used to evaluate how the portfolio might perform under changing economic conditions.

While increases in unemployment and interest rates produced only modest changes in predicted default rates, the exercise demonstrated a practical framework for evaluating portfolio resilience and monitoring future risk.

---

## Business Recommendations

Based on the results, several recommendations can be made:

1. Place greater emphasis on FICO score and debt-to-income ratio during underwriting.
2. Closely monitor borrowers with large loan balances and high debt burdens.
3. Use economic indicators as part of ongoing portfolio monitoring.
4. Develop risk-based lending strategies that account for borrower financial strength and economic conditions.
5. Incorporate stress-testing into regular risk management processes.

---

## Limitations

This project has several limitations:

- The dataset contains fewer credit history variables than those typically available to financial institutions.
- Economic indicators were only available at a monthly level.
- Some macroeconomic variables were highly correlated with one another, which may affect model interpretation.
- Additional variables such as delinquencies, credit utilization, and account history could improve predictive performance.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Conclusion

This project demonstrates how machine learning and economic data can be combined to support credit risk analysis and portfolio monitoring.

Using more than 1.35 million loan records, I developed predictive models to estimate default risk, identified the most important drivers of borrower performance, and created a framework for economic stress testing.

The results showed that borrower financial characteristics—particularly FICO score, income, loan amount, and debt-to-income ratio—were the most influential factors in predicting default risk. While economic conditions provided additional context, individual borrower financial health remained the strongest determinant of loan performance.

This project highlights the value of combining predictive analytics with risk management techniques to support more informed lending decisions.
