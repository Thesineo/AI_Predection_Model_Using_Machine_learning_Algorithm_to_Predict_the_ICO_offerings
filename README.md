# 🪙 AI Model to Predict ICO Success for Fundraising Teams & Startups
### Using Machine Learning Classification Algorithms | Cryptocurrency Offerings

![R](https://img.shields.io/badge/R-4.x-276DC3?logo=r)
![ML](https://img.shields.io/badge/ML-Classification-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Business Problem

Initial Coin Offerings (ICOs) have become a popular crowdfunding mechanism for startups, but their decentralised nature creates high information asymmetry — making it difficult for investors to assess whether a project will successfully reach its fundraising goal. Fraudulent "pump and dump" ICO schemes further compound this risk.

This project builds and compares **6 Machine Learning classification models** to predict the probability of ICO fundraising success, giving startups and investors a data-driven tool to evaluate ICO viability.

- **Y** = Successful fundraising goal achieved  
- **N** = Unsuccessful fundraising goal

---

## 🏗️ Project Pipeline

```
ICO Dataset (2,767 projects)
        ↓
  ┌───────────────────────────────────┐
  │  Data Preprocessing               │
  │  · Missing value imputation       │
  │  · Outlier removal (Boxplot)      │
  │  · Feature encoding               │
  └───────────────────────────────────┘
        ↓
  Train / Test Split (90% / 10%)
  10-Fold Cross Validation
        ↓
  ┌──────┬──────┬──────┬──────┬─────┬──────┐
  │ KNN  │  NB  │  DT  │  RF  │ SVM │  LR  │
  └──────┴──────┴──────┴──────┴─────┴──────┘
        ↓
  Evaluation: Accuracy, AUC, Precision,
              Recall, F-measure, Kappa
```

---

## 📊 Dataset Features

| Feature | Type | Description |
|---|---|---|
| `hasVideo` | Binary (0/1) | Project has a promotional video |
| `hasReddit` | Binary (0/1) | Project has a Reddit community |
| `hasGithub` | Binary (0/1) | Project has a GitHub repository |
| `minInvestment` | Binary (0/1) | Minimum investment threshold set |
| `rating` | Numeric (1–5) | Project rating score |
| `priceUSD` | Numeric | ICO token price in USD |
| `teamSize` | Numeric | Number of team members |
| `coinNum` | Numeric | Number of coins issued |
| `distributedPercentage` | Numeric | % of coins distributed |
| `platform` | Character | Platform used for ICO |
| `success` | Target (Y/N) | Whether fundraising goal was met |

> **Key insight:** Successful projects averaged a rating of **3.39**, vs **2.97** for unsuccessful ones.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `caret` | Model training & cross-validation |
| `class` | KNN classifier |
| `tm` / `e1071` | Naive Bayes classifier |
| `rpart` / `C50` | Decision Tree classifier |
| `randomForest` | Random Forest classifier |
| `pROC` / `ROCR` | ROC curve & AUC evaluation |
| `ggplot2` | Data visualisation |
| `VIM` | Missing value visualisation |

---

## 📦 Setup (R)

```r
install.packages(c("caret", "class", "e1071", "tm", "rpart", "C50",
                   "randomForest", "pROC", "ROCR", "ggplot2", "VIM",
                   "tidyr", "dplyr", "knitr", "mlbench"))
```

```r
# Load dataset
icobench <- read.csv("Downloads/Machinelearningcourseworkright.csv")
str(icobench)
summary(icobench)
```

---

## 🔬 Key Steps

### 1. Missing Value Imputation
```r
library(VIM)
# Impute missing values with column mean
si_icobench$priceUSD[is.na(si_icobench$priceUSD)] <- mean(si_icobench$priceUSD, na.rm = TRUE)
si_icobench$teamSize[is.na(si_icobench$teamSize)] <- mean(si_icobench$teamSize, na.rm = TRUE)
```

### 2. Outlier Removal
```r
# Outliers found in: priceUSD, teamSize, coinNum, distributedPercentage
outlier <- boxplot(data$priceUSD, plot = FALSE)$out
x <- data[-which(data$priceUSD %in% outlier), ]
```

### 3. KNN Model (Best Performer)
```r
library(caret)
K <- 47
knn <- train(success ~ ., data = train, method = "knn",
             preProcess = c("center", "scale"),
             trControl = trainControl(method = "repeatedcv",
                                      number = 10, repeats = 3,
                                      classProbs = TRUE),
             metric = "ROC", tuneLength = 10)
```

### 4. ROC & AUC Evaluation
```r
library(pROC)
roc <- roc(test$success, test_pred)
plot(roc, print.auc = TRUE, auc.polygon = TRUE,
     grid = c(1,2), grid.col = c("green","red"),
     max.auc.polygon = TRUE, auc.polygon.col = "lightblue")
auc(roc)
```

---

## 📈 Model Performance Results

| Classifier | Accuracy (%) | AUC (%) | Precision | Recall | F-Measure | Kappa | Category |
|---|---|---|---|---|---|---|---|
| **KNN** | **100** | **100** | **1.00** | **1.00** | **1.00** | **1.00** | 🥇 Ultimate |
| Decision Tree | 81 | 61 | 0.86 | 0.83 | 0.85 | 0.60 | 🥈 Highline |
| SVM | 71 | 70 | 0.72 | 0.90 | 0.80 | 0.30 | 🥈 Highline |
| Random Forest | 69 | 65 | 0.83 | 0.74 | 0.78 | 0.26 | Baseline |
| Logistic Regression | 66 | 67 | 0.67 | 0.84 | 0.75 | 0.32 | Baseline |
| Naive Bayes | 63 | 70.5 | 0.98 | 0.65 | 0.77 | 0.12 | Baseline |

---

## 💡 Key Findings

- **KNN achieved 100% accuracy** (CCI: 100%, ICI: 0%, AUC: 1.000) — outperforming all other models across every metric by incorporating all dataset attributes including `priceUSD` and `teamSize`
- **DT and SVM** are **Highline classifiers** — solid performers in the 70–85% accuracy range, suitable for simpler use cases
- **NB, RF, and LR** are **Baseline classifiers** — limited because they do not fully leverage numeric attributes to define fundraising success
- **Rating threshold matters** — projects with ratings ≥ 3.3 show significantly higher success probability (visible in Decision Tree splits)
- **Proper pre-processing is critical** — outliers in `priceUSD`, `teamSize`, `coinNum`, and `distributedPercentage` significantly impact model accuracy if not removed

---

## 🗂️ Repository Structure

```
├── data/
│   └── icobench.csv              # ICO project dataset (2,767 records)
├── models/
│   ├── preprocessing.R           # Missing values & outlier handling
│   ├── knn_model.R               # KNN classifier
│   ├── naive_bayes.R             # Naive Bayes classifier
│   ├── decision_tree.R           # Decision Tree (C5.0)
│   ├── random_forest.R           # Random Forest
│   ├── svm_model.R               # Support Vector Machine
│   └── logistic_regression.R     # Logistic Regression
├── outputs/
│   ├── roc_curves/               # ROC plots for all 6 models
│   └── confusion_matrices/       # Confusion matrix outputs
└── README.md
```

---

## ⚠️ Limitations

- KNN's perfect accuracy may reflect overfitting on this specific dataset — further validation on unseen ICO data is recommended
- Models do not incorporate **whitepaper quality**, **social media sentiment**, or **market conditions** at time of ICO launch
- False negatives carry high business risk — a fundraising team predicted as "unsuccessful" may still succeed with external factors

---

## 👤 Author

**Aniket Amar Nerali**
MSc Business Analytics & Decision Sciences — University of Leeds

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Thesineo)
