🎬 Movie Genre Classification using Machine Learning

📌 Project Overview

This project implements a *complete, production-structured Machine Learning pipeline* for *Movie Genre Classification* based on textual movie descriptions.
It is designed with *industry-grade modularity*, *clear separation of concerns*, and *reproducible experimentation*, making it suitable for *HR evaluations, academic reviews, and HOD-level technical discussions*.

The system trains multiple classical ML models, compares their performance, performs error analysis, and supports real-time inference using saved artifacts.

🎯 Key Objectives

* Convert raw movie descriptions into numerical representations using *TF-IDF*
* Train and evaluate *multiple classification models*
* Perform *model comparison and error analysis*
* Persist trained models and vectorizers for reuse
* Provide a *clean inference pipeline*
* Maintain a *scalable, extensible, and professional project structure*

🧠 Machine Learning Models Implemented

The following supervised learning models are trained and evaluated:

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)

Each model is trained independently and saved as a serialized artifact for inference.

🗂️ Project Directory Structure
```
Movie_Genre_Classification/
│
├── artifacts/
│   ├── tfidf.pkl                 # Trained TF-IDF vectorizer
│   ├── logistic.pkl              # Logistic Regression model
│   ├── naive.pkl                 # Naive Bayes model
│   ├── svm.pkl                   # SVM model
│   └── error_analysis.csv        # Misclassification analysis output
│
├── data/
│   ├── description.txt           # Raw movie descriptions
│   ├── train_data.txt            # Training dataset
│   ├── test_data.txt             # Test dataset
│   └── test_data_solution.txt    # Ground truth labels
│
├── eda/
│   └── eda_analysis.py            # Exploratory Data Analysis
│
├── evaluation/
│   ├── model_comparison.py       # Accuracy & performance comparison
│   └── error_analysis.py          # Detailed prediction error analysis
│
├── inference/
│   └── predict.py                # Real-time genre prediction script
│
├── models/
│   ├── train_logistic.py          # Logistic Regression training pipeline
│   ├── train_naive_bayes.py       # Naive Bayes training pipeline
│   └── train_svm.py               # SVM training pipeline
│
├── utils/
│   ├── data_loader.py             # Data loading and preprocessing
│   ├── vectorizer.py              # TF-IDF vectorizer logic
│   └── explainability.py          # Model interpretation utilities
│
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```
🔍 Core Technical Components

1️⃣ Data Processing

* Text data is loaded and cleaned using a centralized utility module.
* Training and test datasets are handled consistently to avoid data leakage.

2️⃣ Feature Engineering

* TF-IDF Vectorization is used to convert text into high-dimensional numerical vectors.
* Vectorizer is trained once and reused during inference.

3️⃣ Model Training

* Each model has its own training script.
* Training logic is modular and reusable.
* Models are persisted using `joblib` for production-style deployment.

4️⃣ Model Evaluation

* Models are compared using accuracy and prediction consistency.
* A dedicated script evaluates and contrasts model performance.

5️⃣ Error Analysis

* Misclassified samples are captured and exported to `error_analysis.csv`.
* Enables deep inspection of model weaknesses and dataset limitations.

6️⃣ Inference Pipeline

* A clean `predict.py` script loads saved artifacts.
* Accepts new movie descriptions and outputs predicted genres.

📊 Explainability & Analysis

* Includes explainability utilities to understand model behavior.
* Supports error-driven improvement and model refinement.
* Suitable for technical interviews and academic defense.

⚙️ Installation & Setup

1️⃣ Create a Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

2️⃣ Install Dependencies

*pip install -r requirements.txt*

▶️ How to Run the Project

Train Models:

*python models/train_logistic.py*
*python models/train_naive_bayes.py*
*python models/train_svm.py*

Compare Models:

*python evaluation/model_comparison.py*

Perform Error Analysis:

*python evaluation/error_analysis.py*

Run Inference:

*python inference/predict.py*

🧪 Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Joblib
* Matplotlib
* TQDM

⭐ Highlights for HR & HOD Review

* Industry-standard *project structure*
* Clear *ML lifecycle separation*
* Multiple models with comparison logic
* Persistent artifacts for deployment
* Strong emphasis on *error analysis*
* Clean, readable, and maintainable codebase
* Suitable for *ML internships, final-year projects, and interviews*

🚀 Future Enhancements

* Deep Learning models (LSTM / Transformers)
* REST API deployment (Flask/FastAPI)
* Web-based UI for predictions
* Advanced explainability (SHAP, LIME)

📌 Author

*Shree Abiraami M*
Machine Learning & Software Engineering Enthusiast
