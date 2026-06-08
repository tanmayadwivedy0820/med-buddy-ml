# 🩺 Med Buddy – Production-Ready ML Healthcare Prediction System

Med Buddy is an end-to-end Machine Learning application that predicts the probability of a medical outcome using validated ML techniques and industry-standard engineering practices.  
The project demonstrates the complete ML lifecycle — from exploratory data analysis and model benchmarking to production deployment on AWS.

---

## 🚀 Key Features

- End-to-end ML pipeline: **EDA → Modeling → Validation → Deployment**
- Exploratory Data Analysis with outlier detection
- Robust cross-validation to prevent data leakage
- Baseline and advanced model benchmarking
- Medical domain–aware evaluation metrics
- Notebook-to-production code transition
- **Multipage Streamlit dashboard** (prediction, EDA explorer, model insights)
- Full-stack ML deployment on AWS EC2

---

## 🔍 Problem Approach

In healthcare prediction systems, minimizing **false negatives** is critical.  
Therefore, this project prioritizes **Recall** and **F1-score** over accuracy to ensure reliable and safer predictions, especially in the presence of class imbalance.

---

## 📊 Exploratory Data Analysis (EDA)

- Performed data visualization using **Matplotlib** and **Seaborn**
- Analyzed feature distributions and relationships
- Identified and examined **outliers** using boxplots and distribution plots
- Insights from EDA guided feature selection and model choice

---

## 🧠 Machine Learning Methodology

### Validation Strategy
To ensure unbiased and generalizable model performance:
- **StratifiedKFold** to preserve class distribution across folds  
- **GroupShuffleSplit** to prevent data leakage across grouped samples  

### Model Benchmarking
A structured comparison pipeline was implemented using:
- Logistic Regression (**baseline model**)
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

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

## 🏗️ System Architecture

The application is split into two decoupled layers communicating over HTTP.

**Backend (FastAPI)** — owns the model and data:
- `/predict-heart-disease` — runs inference on patient input
- `/dataset-stats` — read-only EDA data (summary stats, distributions, class balance)
- `/model-info` — model configuration and feature importances
- The trained model (a `StandardScaler → RandomForestClassifier` pipeline) is loaded once at startup as a module-level singleton and shared across endpoints.

**Frontend (Streamlit, multipage)** — pure presentation, no ML logic:
- Built with the `st.navigation` / `st.Page` API; `app.py` is a thin router.
- `config.py` holds a single `API_BASE_URL`; each page composes its own endpoint.
- Three pages: **Predictor** (form → API), **Dataset Explorer** (EDA dashboard), **Model Insights** (feature importances + model config).
- API responses are cached with `st.cache_data` to avoid redundant calls on rerun.

The frontend never touches the CSV or the model artifact directly — everything goes through the API. Swapping the model (e.g. RandomForest → XGBoost) requires no frontend changes.

---

## 🛠️ Tech Stack

**Machine Learning & EDA:** Python, Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn  
**Backend:** FastAPI, Uvicorn  
**Frontend:** Streamlit  
**Deployment:** AWS EC2 (Ubuntu), Git, GitHub  

---

## 📂 Project Structure

```bash
.
├── backend/              # FastAPI backend (API + inference logic)
├── frontend/             # Streamlit multipage frontend
│   ├── app.py            # navigation router
│   ├── config.py         # shared API config
│   └── views/            # individual pages
├── dataset/              # dataset (heart.csv)
├── model_dir/            # trained model artifact
├── notebook_files/       # EDA & experimentation notebooks
├── requirements.txt
├── env_template.txt
├── README.md
└── .gitignore
```

---

## ▶️ Running the Application Locally

**Backend** (from project root):
```bash
uvicorn backend.main:app --reload
```

**Frontend** (from project root, in a second terminal):
```bash
streamlit run frontend/app.py
```

---

## ☁️ Deployment

The application was deployed and validated on **AWS EC2 (Ubuntu)** to confirm the
full stack runs correctly on public infrastructure. The instance has since been
stopped to avoid idle hosting costs — it can be redeployed in minutes.

Deployment setup:
- FastAPI backend served via Uvicorn
- Streamlit frontend exposed over the instance's public IP
- Security groups configured to allow the required ports
- Environment-based configuration for production readiness

> The instance is not kept running continuously, as a portfolio demo does not
> justify an always-on paid server. This is a deliberate cost decision, not a deployment gap.

---

## 🎯 Why This Project Stands Out

- Uses robust cross-validation techniques
- Demonstrates baseline vs advanced model comparison
- Applies domain-relevant performance metrics
- Shows transition from experimentation to production
- Clean frontend/backend separation via a documented API contract
- Implements full-stack ML deployment on the cloud

---

## 🔮 Future Enhancements

- User authentication
- Dockerization and CI/CD pipeline
- Model monitoring and drift detection
- SHAP / permutation-based feature importance
- Scalable cloud-native deployment