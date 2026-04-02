# 📊 Telco Customer Churn Prediction (MLOps Project)

## 🚀 Project Overview

This project is an end-to-end **Machine Learning + MLOps pipeline** that predicts whether a customer will churn or not.

It includes:

* Data preprocessing
* Model training
* Experiment tracking using MLflow
* Model deployment using Flask
* Docker containerization
* CI/CD pipeline using GitHub Actions

---

## 🎯 Objective

To build a **production-ready ML system** that can:

* Train a churn prediction model
* Track experiments
* Serve predictions through an API
* Automate build and testing using CI/CD

---

## 🧱 Project Structure

```
TELCO-CUSTOMER-CHURN/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   ├── model_bundle.pkl
│   └── columns.pkl
│
├── notebooks/
│
├── src/
│   ├── app.py
│   ├── train.py
│   ├── predict.py
│   └── data_preprocessing.py
│
├── templates/
│   └── index.html
│
├── mlruns/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* **Python**
* **Pandas, NumPy, Scikit-learn**
* **Flask (API)**
* **MLflow (Experiment Tracking)**
* **Docker (Containerization)**
* **GitHub Actions (CI/CD)**

---

## 🔄 Workflow

```
Data → Preprocessing → Training → MLflow Tracking 
→ Model Saving → Flask API → Docker → CI/CD
```

---

## 🧠 Model Details

* Problem Type: Classification
* Target Variable: `Churn`
* Algorithms: (e.g., Logistic Regression / Random Forest)
* Evaluation Metrics: Accuracy, Precision, Recall

---

## 📈 MLflow Tracking

* Tracks experiments and parameters
* Logs metrics and models
* Enables comparison of multiple runs

Run MLflow locally:

```
mlflow ui --port 5002
```

---

## 🌐 Running the Application (Locally)

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run the Flask app

```
python src/app.py
```

### 3. Open in browser

```
http://127.0.0.1:5000
```

---

## 🐳 Docker Setup

### Build image

```
docker build -t churn-app .
```

### Run container

```
docker run -p 5000:5000 churn-app
```

---

## ⚡ CI/CD Pipeline

* Triggered on every push
* Steps:

  * Install dependencies
  * Run basic test
  * Build Docker image

Workflow file:

```
.github/workflows/ci-cd.yml
```

---

## 🔥 Key Features

✅ End-to-end ML pipeline
✅ Experiment tracking with MLflow
✅ Model + feature consistency using `columns.pkl`
✅ Flask-based prediction API
✅ Dockerized deployment
✅ Automated CI/CD pipeline

---

## 📌 Future Improvements

* Add model versioning (MLflow Registry)
* Add logging & monitoring
* Deploy to cloud (AWS / Render / Railway)
* Add unit testing
* Add input validation

---

## 🙋‍♂️ Author

**Manab Das**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
