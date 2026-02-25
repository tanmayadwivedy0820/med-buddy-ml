# 🩺 Med Buddy – Production-Ready ML Healthcare Prediction System

Med Buddy is an end-to-end Machine Learning application that predicts the probability of a medical outcome using validated ML techniques and industry-standard engineering practices.  
The project demonstrates the complete ML lifecycle — from experimentation and model selection to production deployment on AWS.

---

## 🚀 Key Features

- End-to-end ML pipeline: **EDA → Modeling → Validation → Deployment**
- Robust cross-validation to prevent data leakage
- Baseline and advanced model benchmarking
- Medical domain–aware evaluation metrics
- Notebook-to-production code transition
- Full-stack ML deployment on AWS EC2

---

## 🔍 Problem Approach

In healthcare prediction systems, minimizing **false negatives** is critical.  
Therefore, this project prioritizes **Recall** and **F1-score** over accuracy to ensure reliable and safer predictions, especially in the presence of class imbalance.

---

## 🧠 Machine Learning Methodology

### Validation Strategy
To ensure unbiased and generalizable model performance:
- **StratifiedKFold** to preserve class distribution across folds  
- **GroupShuffleSplit** to prevent data leakage across grouped samples  

---

### Model Benchmarking

A structured comparison pipeline was implemented using:

- Logistic Regression (**baseline model**)
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

---

### Evaluation Metrics

- **Recall** – prioritized to reduce false negatives  
- **F1-score** – balances precision and recall  

Accuracy was intentionally avoided due to class imbalance and medical risk sensitivity.

---

## 🏆 Final Model Selection

- **Selected Model:** Random Forest  
- Chosen based on:
  - Highest Recall
  - Strong F1-score
  - Stable cross-validation performance  
- Hyperparameter tuning applied to obtain optimal parameters  
- Final model serialized for production inference  

---

## ⚙️ Productionization
- Experimental Jupyter notebooks converted into **modular Python scripts**
- Clear separation of:
  - Training pipeline
  - Inference logic
  - API serving layer  
- Reproducible and scalable codebase suitable for deployment  

---

## 🏗️ System Architecture

**Streamlit Frontend → FastAPI Backend → Trained ML Model → Prediction Response**

---

## 🛠️ Tech Stack
**Machine Learning:** Python, Scikit-learn, XGBoost, Pandas, NumPy 
**Backend:** FastAPI, Uvicorn  
**Frontend:** Streamlit  
**Deployment:** AWS EC2 (Ubuntu), Git, GitHub  

## 📂 Project Structure
.
├── backend/              # FastAPI backend
├── frontend/             # Streamlit frontend
├── dataset/              # Raw & processed datasets
├── model_dir/            # Trained model & artifacts
├── notebook_files/       # EDA & experimentation notebooks
├── requirements.txt
├── env_template.txt
├── README.md
└── .gitignore

▶️ Running the Application Locally
Backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
Frontend
cd frontend
streamlit run app.py

☁️ Deployment
Deployed on AWS EC2 (Ubuntu)
FastAPI served using Uvicorn
Streamlit frontend exposed via public EC2 IP
Security groups configured for required ports
Environment-based configuration for production readiness

🎯 Why This Project Stands Out
Uses robust cross-validation techniques
Demonstrates baseline vs advanced model comparison
Applies domain-relevant performance metrics
Shows transition from experimentation to production
Implements full-stack ML deployment on the cloud

🔮 Future Enhancements
Addition of user Authentication
Dockerization and CI/CD pipeline
Model monitoring and drift detection
Enhanced feature engineering
Scalable cloud-native deployment
