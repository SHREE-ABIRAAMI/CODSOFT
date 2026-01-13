---

# CODSOFT – Machine Learning Internship Projects

This repository contains a curated set of **Machine Learning and Data Science projects** developed as part of the **CODSOFT Machine Learning Internship Program**.

Each task is implemented as an **independent, end-to-end ML project**, following **industry-aligned engineering practices** such as modular design, reproducibility, rigorous evaluation, and explainability.
The repository demonstrates progressive skill development across **NLP, structured data modeling, and real-world ML workflows**.

---

## Repository Structure

```
CODSOFT/
│
├── Movie_Genre_Classification/        # Task 1 (Completed)
│   ├── data/
│   ├── eda/
│   ├── models/
│   ├── evaluation/
│   ├── inference/
│   ├── utils/
|   └── requirements.txt
│
├── Credit_Card_Fraud_Detection/       # Task 2 (Completed)
│   ├── data/
│   ├── artifacts/
│   ├── utils/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── requirements.txt
│
├── Spam_SMS_Detection/                # Task 3 (Completed)
│   ├── data/
│   ├── utils/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── artifacts/
│   └── requirements.txt
│
└── README.md
```

---

## ✅ Completed Internship Tasks

### **Task 1: Movie Genre Classification**

An end-to-end **NLP-based multi-class classification system** that predicts movie genres from plot summaries.

**Key Highlights**

* TF-IDF feature extraction with uni-grams and bi-grams
* Multiple model training and comparison:

  * Naive Bayes
  * Logistic Regression
  * Calibrated Linear SVM
* Top-K genre prediction with confidence scores
* Explainable AI using linear model coefficients
* Detailed error analysis and confusion matrix visualization

---

### **Task 2: Credit Card Fraud Detection**

A complete **binary classification system** for detecting fraudulent credit card transactions using structured transactional data.

**Key Highlights**

* Robust data preprocessing:

  * StandardScaler for numerical features
  * OneHotEncoder for categorical features
* Class imbalance handling:

  * Stratified sampling
  * Class-weighted models
* Model training and comparison:

  * Logistic Regression (interpretable baseline)
  * Random Forest (non-linear ensemble)
* Comprehensive evaluation:

  * Precision, Recall, F1-score
  * ROC-AUC analysis
* Explainability module identifying **top contributing features**
* Error analysis pipeline capturing:

  * False positives
  * False negatives
* Production-style inference pipeline for unseen data

---

### **Task 3: Spam SMS Detection**

A production-oriented **text classification system** for detecting spam SMS messages using classical NLP and machine learning techniques.

**Key Highlights**

* Text preprocessing pipeline:

  * Lowercasing
  * URL, number, and punctuation removal
* TF-IDF vectorization with uni-grams and bi-grams
* Model training and comparison:

  * Naive Bayes
  * Logistic Regression
  * Linear SVM (final selected model)
* Detailed performance evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* Error analysis module saving misclassified messages for inspection
* Command-line inference script for real-time SMS prediction
* Reusable vectorizer and model artifacts using Joblib

---

## 🧠 Technical Stack

* **Programming Language:** Python
* **Core Libraries:** NumPy, Pandas
* **Machine Learning:** Scikit-learn
* **NLP:** TF-IDF Vectorization
* **Model Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC
* **Explainability:** Model coefficient–based feature analysis
* **Model Persistence:** Joblib

---

## 📈 Engineering Practices Followed

* Modular and scalable repository structure
* Clear separation of concerns:

  * Data loading
  * Feature engineering
  * Model training
  * Evaluation
  * Inference
* Reproducible experiments with saved artifacts
* Explainability-first approach for model decisions
* Clean, readable, and well-documented code
* Production-style inference workflows
* Repository designed to support multiple independent ML projects

---

## 🎯 Internship Completion Status

✔️ **All 3 tasks of the CODSOFT Machine Learning Internship have been successfully completed**
✔️ Each task follows a professional, end-to-end ML pipeline
✔️ The repository reflects real-world ML engineering standards rather than academic demos

---

## 👤 Author

*Shree Abiraami M*
Machine Learning Engineer & Python Developer
CODSOFT Machine Learning Intern

---

## 📄 License

This repository is created for **educational and internship evaluation purposes**.
The codebase is structured to reflect **production-oriented machine learning workflows** and best practices.

---
