
---

# CODSOFT – Machine Learning Internship Projects

This repository contains a curated set of **Machine Learning and Data Science projects** developed as part of the **CODSOFT Internship Program**.
Each task is implemented as an **independent, end-to-end project**, following industry-aligned practices such as modular design, reproducibility, evaluation rigor, and explainability.

The repository is structured to scale cleanly as new tasks are added, while maintaining clarity and separation of concerns.

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
│   └── README.md
│
├── Credit_Card_Fraud_Detection/       # Task 2 (Completed)
│   ├── data/
│   ├── artifacts/
│   ├── utils/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── requirements.txt
│   └── README.md
│
├── Task_3/                            # Planned
├── Task_4/                            # Planned
├── Task_5/                            # Planned
│
└── README.md
```

---

## ✅ Completed Tasks

### **Task 1: Movie Genre Classification**

An end-to-end **NLP and Machine Learning pipeline** that predicts movie genres based on plot summaries.

**Key Highlights**

* TF-IDF based feature extraction with n-grams
* Multiple model training and comparison:

  * Naive Bayes
  * Logistic Regression
  * Calibrated Linear SVM
* Top-K genre prediction with confidence scores
* Explainable AI using linear model coefficients
* Detailed error analysis and confusion matrix visualization

Detailed documentation is available in:
`Movie_Genre_Classification/README.md`

---

### **Task 2: Credit Card Fraud Detection**

A complete **binary classification system** designed to detect fraudulent credit card transactions using structured transaction data.

**Key Highlights**

* Robust data preprocessing using:

  * StandardScaler for numerical features
  * OneHotEncoder for categorical features
* Handling class imbalance through:

  * Stratified sampling
  * Class-weighted models
* Model training and comparison:

  * Logistic Regression (interpretable baseline)
  * Random Forest (non-linear ensemble model)
* Comprehensive evaluation:

  * Classification report
  * ROC-AUC score
* Explainability module that identifies **top contributing features** behind individual predictions
* Error analysis pipeline generating:

  * Correct predictions
  * False positives
  * False negatives (saved for post-model diagnostics)
* Production-style inference script simulating real-world predictions on unseen data

Detailed documentation is available in:
`Credit_Card_Fraud_Detection/README.md`

---

## 🧠 Technical Stack

* **Programming Language:** Python
* **Core Libraries:** NumPy, Pandas
* **Machine Learning:** Scikit-learn
* **Data Preprocessing:** StandardScaler, OneHotEncoder
* **Model Evaluation:** Classification metrics, ROC-AUC
* **Explainability:** Model coefficient-based feature contribution analysis
* **Model Persistence:** Joblib

---

## 📈 Engineering Practices Followed

* Modular and scalable project structure
* Clear separation of concerns:

  * Data loading
  * Feature engineering
  * Training
  * Evaluation
  * Inference
* Reproducible experiments with saved artifacts
* Explainability-first approach for model decisions
* Clean, readable, and well-documented code
* Repository layout designed for multi-project growth

---

## 🚀 Upcoming Tasks

The following tasks will be added incrementally as part of the internship program:

* **Task 3:** Planned
* **Task 4:** Planned
* **Task 5:** Planned

Each task will maintain the same professional structure, evaluation depth, and documentation standard followed in Tasks 1 and 2.

---

## 👤 Author

*Shree Abiraami M*
Machine Learning Engineer & Python Developer
CODSOFT ML Intern

---

## 📄 License

This repository is created for **educational and internship evaluation purposes**.
The codebase is modular, well-documented, and structured to reflect production-oriented ML workflows.

---