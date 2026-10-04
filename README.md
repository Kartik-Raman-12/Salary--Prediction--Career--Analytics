# Salary Prediction & Career Analytics

An end-to-end machine learning project for predicting salary using regression techniques, statistical analysis, feature engineering, and model diagnostics.

The project compares multiple regression approaches and also implements Linear Regression from scratch using both the **Normal Equation** and **Gradient Descent**.

---

## 📌 Project Overview

Salary prediction is a regression problem where the objective is to estimate an individual's salary based on relevant career and professional attributes.

This project develops a complete regression pipeline covering:

- Data preprocessing and validation
- Exploratory Data Analysis (EDA)
- Feature engineering
- Categorical variable encoding
- Multiple regression techniques
- Regularization
- Cross-validation
- Statistical inference using Statsmodels
- Regression diagnostics
- Linear Regression implementation from scratch
- Numerical stability and convergence analysis
- Model comparison

The project focuses not only on predictive performance but also on understanding the **statistical assumptions and numerical behavior of regression models**.

---

## 🎯 Objectives

The main objectives of the project are:

1. Build an end-to-end salary prediction pipeline.
2. Explore relationships between career attributes and salary.
3. Compare different regression models.
4. Evaluate models using appropriate regression metrics.
5. Analyze multicollinearity and regression assumptions.
6. Perform statistical inference using Statsmodels.
7. Implement Linear Regression from scratch.
8. Compare analytical and iterative optimization methods.
9. Study numerical stability and convergence of regression algorithms.

---

## 📊 Dataset

After preprocessing and cleaning, the dataset contains:

- **1,784 records**
- **187 transformed/engineered features**

The preprocessing pipeline includes handling categorical variables, feature transformation, duplicate checking, outlier analysis, and preparation of the final modeling matrix.

> **Note:** The exact dataset source and original feature descriptions should be added here if the dataset is externally sourced.

---

## 🔬 Project Workflow

```text
Raw Dataset
     │
     ▼
Data Validation & Cleaning
     │
     ▼
Exploratory Data Analysis
     │
     ├── Missing Values
     ├── Duplicate Records
     ├── Outlier Analysis
     ├── Distribution Analysis
     └── Correlation Analysis
     │
     ▼
Feature Engineering
     │
     ├── Numerical Features
     ├── Categorical Encoding
     └── Feature Transformation
     │
     ▼
Train/Test Split
     │
     ▼
Regression Models
     │
     ├── Linear Regression
     ├── Polynomial Regression
     ├── Ridge Regression
     └── Lasso Regression
     │
     ▼
Cross-Validation & Evaluation
     │
     ▼
Statistical Analysis
     │
     ├── OLS
     ├── VIF
     ├── Breusch–Pagan Test
     ├── Robust Standard Errors
     └── Residual & Leverage Diagnostics
     │
     ▼
Regression From Scratch
     │
     ├── Normal Equation
     └── Gradient Descent
     │
     ▼
Final Model Comparison
```

---

## 🤖 Models Implemented

### 1. Linear Regression

A multiple linear regression model is used as the baseline predictive model.

The model assumes:

$\[
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_px_p + \epsilon
\]$

where:

- $\(y\)$ = salary
- $\(x_i\)$ = predictor variables
- $\(\beta_i\)$ = regression coefficients
- $\(\epsilon\)$ = error term

---

### 2. Polynomial Regression

Polynomial features are introduced to model nonlinear relationships between predictors and salary.

The polynomial model allows the regression function to capture nonlinear patterns while still using a linear regression estimator.

---

### 3. Ridge Regression

Ridge regression introduces an L2 regularization penalty:

$$\[
\min_{\beta}
\left[
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
+
\lambda\sum_{j=1}^{p}\beta_j^2
\right]
\]$$

This helps control coefficient magnitude and reduce the effect of multicollinearity.

---

### 4. Lasso Regression

Lasso uses L1 regularization:

$$\[
\min_{\beta}
\left[
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
+
\lambda\sum_{j=1}^{p}|\beta_j|
\right]
\]$$

The L1 penalty can drive some coefficients to zero, providing a form of feature selection.

---

## 📈 Model Evaluation

The models are evaluated using standard regression metrics:

### R² Score

$\[
R^2 =
1-
\frac{\sum(y_i-\hat y_i)^2}
{\sum(y_i-\bar y)^2}
\]$

### Root Mean Squared Error

$\[
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}(y_i-\hat y_i)^2
}
\]$

### Cross-Validation

**5-fold cross-validation** is used to evaluate model generalization and reduce dependence on a single train/test split.

---

## 🏆 Results

The best-performing model in the notebook was **Polynomial Regression**.

| Metric | Result |
|---|---:|
| Test R² | **0.885** |
| Test RMSE | **₹18.3K** |
| 5-Fold CV R² | **0.871** |
| Training Records | **1,784** |
| Transformed Features | **187** |

The cross-validation result indicates that the model maintains strong predictive performance across different training/validation splits rather than relying solely on a single test split.

---

## 📊 Statistical Analysis with Statsmodels

In addition to predictive modeling with Scikit-learn, the project uses **Statsmodels** for statistical analysis of the regression model.

The analysis includes:

### Ordinary Least Squares (OLS)

OLS is used to examine:

- Regression coefficients
- Standard errors
- Statistical significance
- Confidence intervals
- Model-level statistics

### Multicollinearity

Variance Inflation Factor (VIF) is examined to identify potential multicollinearity among predictors.

### Heteroscedasticity

The **Breusch–Pagan test** is used to investigate whether the variance of regression residuals is constant.

### Robust Standard Errors

Robust covariance estimates are considered to make statistical inference less sensitive to heteroscedasticity.

### Residual Diagnostics

Residual analysis is performed to investigate:

- Residual distributions
- Prediction errors
- Potential systematic patterns
- Model assumptions

### Leverage Diagnostics

Leverage analysis is used to identify observations that may have disproportionate influence on the fitted regression model.

---

## 🧮 Linear Regression From Scratch

To understand the underlying mathematics, Linear Regression is implemented without relying entirely on a high-level regression estimator.

### Normal Equation

The closed-form solution is:

$$\[ \hat{\beta} = (X^TX)^{-1}X^Ty \]$$

The implementation also investigates numerical stability when solving the regression system.

### Gradient Descent

The parameters are optimized iteratively by minimizing Mean Squared Error.

The parameter update follows:

$\[
\beta_j
\leftarrow
\beta_j-\alpha
\frac{\partial J}{\partial\beta_j}
\]$

where:

- $\(\alpha\)$ = learning rate
- $\(J\)$ = loss function
- $\(\beta_j\)$ = model parameter

The convergence behavior is analyzed using the loss across iterations.

---

## 🔎 Numerical Stability

The project also investigates numerical issues that can occur when solving regression problems with a large number of transformed features.

The analysis includes:

- Matrix conditioning
- Rank deficiency
- Stable least-squares solutions
- Comparison of analytical and iterative optimization
- Convergence behavior

This provides a deeper understanding of why mathematically equivalent regression implementations can behave differently in practical numerical computation.

---

## 🛠️ Technologies Used

### Programming

- Python

### Data Processing

- NumPy
- Pandas

### Machine Learning

- Scikit-learn

### Statistical Modeling

- Statsmodels

### Visualization

- Matplotlib

### Environment

- Google Colab / Jupyter Notebook

---

## 📁 Project Structure

```text
Salary-Prediction-Career-Analytics/
│
├── README.md
│
├── Salary_Prediction_Career_Analytics_Phase1.ipynb
│
├── data/
│   └── salary_dataset.csv
│
└── figures/
    └── ...
```

> Dataset and generated figures can be added to the repository depending on dataset licensing and repository size.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/salary-prediction-career-analytics.git
cd salary-prediction-career-analytics
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Salary_Prediction_Career_Analytics_Phase1.ipynb
```

Alternatively, the notebook can be executed directly in **Google Colab**.

---

## 📌 Key Takeaways

- Polynomial regression provided the strongest predictive performance among the evaluated models.
- Regularization with Ridge and Lasso provides useful approaches for controlling model complexity.
- Cross-validation provides a more reliable estimate of model generalization.
- Statistical diagnostics reveal important considerations beyond predictive accuracy.
- Implementing Linear Regression from scratch provides insight into the mathematical and numerical foundations of regression.
- Numerical conditioning becomes particularly important when working with a large engineered feature matrix.

---

## 🔮 Future Improvements

Potential extensions include:

- Hyperparameter optimization using GridSearchCV/RandomizedSearchCV
- More extensive feature selection
- Alternative nonlinear regression models
- Tree-based ensemble models
- Explainable ML techniques
- Prediction intervals for salary estimates
- Deployment using Streamlit or Flask
- Interactive career/salary analytics dashboard
- Incorporating additional real-world salary and job-market data

---

## 👨‍💻 Author

**Kartik Raman**

---

## ⭐ Project Focus

This project was developed to strengthen practical understanding of:

**Regression → Statistical Inference → Model Diagnostics → Numerical Optimization → Machine Learning**

If you find the project useful, consider giving the repository a ⭐.
