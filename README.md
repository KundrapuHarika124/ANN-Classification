# ANN-Classification

---

## 🎯 Goal

Build an ANN classifier that predicts if a customer will **exit the bank** based on their demographic and account info.

**Target column:** `Exited`  
- `0` → Still a customer  
- `1` → Left the bank

---

## 📌 Features Used

| Column Name         | Description                           |
|---------------------|---------------------------------------|
| `CreditScore`       | Customer's credit score               |
| `Geography`         | Country: France, Germany, Spain       |
| `Gender`            | Male or Female                        |
| `Age`               | Customer's age                        |
| `Tenure`            | No. of years with the bank            |
| `Balance`           | Account balance                       |
| `NumOfProducts`     | No. of bank products used             |
| `HasCrCard`         | 1 if has a credit card, else 0        |
| `IsActiveMember`    | 1 if active, else 0                   |
| `EstimatedSalary`   | Predicted salary                      |
| `Exited`            | Target label (1 = churn, 0 = stay)    |

---

## 🔧 How It Works

### 📘 `experiments.ipynb`
- Data cleaning and preprocessing
- One-hot encoding & scaling
- ANN architecture (2 hidden layers + output sigmoid)
- Training + evaluation

### 🎯 `prediction.ipynb`
- Load `model.keras` and scalers
- Take new input data
- Preprocess → Predict → Output result

### 🧠 `model.keras`
- Pretrained ANN with 80–85% accuracy

### 🧪 Saved Transformers
- `scaler.pkl` → For scaling test data  
- `label_encoder.pkl` & `onehot_encoder.pkl` → To encode input like training



