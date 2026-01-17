# 🚀 Churn Prediction System

A production-ready, full-stack AI application for predicting customer churn in the telecommunications industry. This system uses a **LightGBM** model served via **FastAPI** with a modern, responsive **HTML/CSS/JS** frontend.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-96%25-brightgreen)

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Workflow](#-workflow)

---

## 📖 Overview

Customer churn is a critical metric for telecom companies. This project provides an end-to-end solution to predict whether a customer is likely to leave (churn) based on their usage patterns, demographics, and account status.

The system consists of:
1.  **Training Pipeline**: Fetches data, trains a LightGBM classifier, and saves artifacts.
2.  **Backend API**: A high-performance FastAPI server that loads the model and serves predictions.
3.  **Frontend UI**: A clean, glassmorphism-styled web interface for easy interaction.

---

## ✨ Features

- **High Accuracy**: ~96% accuracy on the test set.
- **Real-time Inference**: Low-latency predictions (<50ms).
- **Batch Processing**: Support for bulk predictions.
- **Interactive UI**: User-friendly web form with instant feedback.
- **Risk Assessment**: Categorizes customers into Low, Medium, or High risk.
- **Dockerized**: Easy deployment with Docker Compose.

---

## 🛠 Tech Stack

- **Language**: Python 3.11+
- **ML Framework**: LightGBM, Scikit-learn, Pandas
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript (Vanilla)
- **Containerization**: Docker, Docker Compose

---

## 📊 Dataset

We use the **Iranian Churn Dataset** (UCI ID: 563).
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)
- **Size**: 3150 rows, 13 features.
- **Target**: `Churn` (0: Non-churn, 1: Churn).

**Key Features:**
- `Call_Failure`: Number of call failures.
- `Seconds_of_Use`: Total usage time.
- `Subscription_Length`: Months of service.
- `Age_Group`: Customer age category.
- `Status`: Active/Non-active.

---

## 📂 Project Structure

```
project1/
├── backend/                # FastAPI Backend
│   ├── main.py             # API Entry point
│   ├── models.py           # Pydantic Data Models
│   ├── requirements.txt    # Python Dependencies
│   ├── Dockerfile          # Backend Docker Image
│   └── tests/              # Pytest Suite
├── frontend/               # Static Frontend
│   ├── index.html          # Main UI
│   ├── style.css           # Styling (Glassmorphism)
│   └── script.js           # Frontend Logic
├── docs/                   # Documentation
│   ├── DATASET_INFO.md     # Dataset Details
│   ├── WORKFLOW_AND_IO.md  # Architecture & I/O
│   └── CLIENT_PRESENTATION.md # Business Slide Content
├── train.py                # Model Training Script
├── docker-compose.yml      # Docker Orchestration
└── README.md               # This file
```

---

## ⚡ Installation & Setup

### Option 1: Docker (Recommended)

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd project1
    ```

2.  **Run with Docker Compose**:
    ```bash
    docker-compose up --build -d
    ```

3.  **Access the App**:
    - Frontend: [http://localhost:8000](http://localhost:8000)
    - API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Local Development

1.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r backend/requirements.txt
    ```

3.  **Train the Model**:
    ```bash
    python train.py
    ```
    *This will generate `.pkl` files in the `backend/` directory.*

4.  **Run the Backend**:
    ```bash
    uvicorn backend.main:app --reload
    ```

5.  **Run the Frontend** (in a separate terminal):
    ```bash
    cd frontend
    python3 -m http.server 8080
    ```
    Access at [http://localhost:8080](http://localhost:8080).

---

## 🎮 Usage

1.  Open the web interface.
2.  Fill in the customer details (Age, Usage, Plan, etc.).
3.  Click **"Analyze Risk"**.
4.  View the prediction result, probability score, and risk level.

---

## 🔌 API Documentation

### `POST /predict`
Predict churn for a single customer.

**Request Body:**
```json
{
  "Call_Failure": 8,
  "Complains": 0,
  "Subscription_Length": 38,
  "Charge_Amount": 0,
  "Seconds_of_Use": 4370,
  "Frequency_of_use": 71,
  "Frequency_of_SMS": 5,
  "Distinct_Called_Numbers": 17,
  "Age_Group": 3,
  "Tariff_Plan": 1,
  "Status": 1,
  "Age": 30,
  "Customer_Value": 197.64
}
```

**Response:**
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.042,
  "risk_level": "Low",
  "confidence": 0.958
}
```

### `GET /health`
Check system status and model accuracy.

---

## 🔄 Workflow

1.  **Data Ingestion**: `train.py` downloads data from UCI.
2.  **Preprocessing**: Data is cleaned and split (80/20).
3.  **Training**: LightGBM model is trained and validated (10-fold CV).
4.  **Serving**: FastAPI loads the model artifacts on startup.
5.  **Inference**: User requests are validated and passed to the model for prediction.
