# Student Grade & Performance Predictor

A machine learning system that predicts a student's exam score (0–100) and assigns a letter grade based on academic and behavioral inputs, deployed as a Flask REST API and integrated into a live React web application.

---

## Background

The initial phase of this project was a **collaborative work done as part of CSE422 – Artificial Intelligence**, completed together with a peer. The goal was to explore the different factors affecting student academic performance using machine learning techniques.

The initial work is in [`project_code_initial.ipynb`](./project_code_initial.ipynb) and focused on **classification analysis** — continuous exam scores were converted into discrete grade categories to allow comparison of classification models:

| Model | Accuracy |
|---|---|
| Logistic Regression | 98% |
| Naive Bayes | 92% |
| KNN | 83% |

The project was also **expanded to include regression analysis**, predicting exact numerical scores using Linear Regression and KNN algorithms.

All subsequent work — including the Flask REST API, preprocessing pipeline for deployment, model serialization, and the web integration — was **independently implemented by me**.

---

## What Changed from Notebook → Deployment

| Aspect | Jupyter Notebook (Group Work) | Flask Deployment (Solo) |
|---|---|---|
| Task | Classification (grade categories) | Regression (exact score 0–100) |
| Models deployed | — | Linear Regression, KNN Regressor |
| Prediction | Offline notebook | Live REST API with averaged predictions |
| Interface | None | React web page — users fill a form and get instant results |
| Output | Grade category | Numerical score + letter grade |

---

## Dataset

- **Source:** [Kaggle — Exam Score Prediction Dataset](https://www.kaggle.com/datasets/kundanbedmutha/exam-score-prediction-dataset)
- **Size:** ~20,000 student records
- **Target:** `exam_score` (continuous, 0–100)
- **Note:** The dataset file (`Exam_Score_Prediction.csv`) is **not included** in this repo. Download it from the link above and place it in this directory before running `train_and_save.py`.

### Features (11 inputs)

| Feature | Type | Values |
|---|---|---|
| `age` | Numeric | Student age |
| `gender` | Categorical | `male`, `female`, `other` |
| `course` | Categorical | `diploma`, `bca`, `b.sc`, etc. |
| `study_hours` | Numeric | Daily study hours |
| `class_attendance` | Numeric | Attendance percentage |
| `internet_access` | Binary | `yes`, `no` |
| `sleep_hours` | Numeric | Hours of sleep per night |
| `sleep_quality` | Ordinal | `poor`, `average`, `good` |
| `study_method` | Categorical | `coaching`, `online videos`, etc. |
| `facility_rating` | Ordinal | `low`, `medium`, `high` |
| `exam_difficulty` | Ordinal | `easy`, `moderate`, `hard` |

---

## Preprocessing

Applied identically during training and at inference time:

1. **Drop `student_id`** — not a predictive feature
2. **Label encode** ordinal and binary columns:
   - `gender` → `{male: 0, female: 1, other: 2}`
   - `internet_access` → `{no: 0, yes: 1}`
   - `sleep_quality` → `{poor: 0, average: 1, good: 2}`
   - `facility_rating` → `{low: 0, medium: 1, high: 2}`
   - `exam_difficulty` → `{easy: 0, moderate: 1, hard: 2}`
3. **One-hot encode** nominal columns: `course`, `study_method`
4. **StandardScaler** — fit on training data, applied to all inputs
5. **80/20 train/test split** (`random_state=42`)

---

## Models

| Model | Algorithm | Key Hyperparameters | Saved As |
|---|---|---|---|
| Linear Regression | `sklearn.linear_model.LinearRegression` | defaults (OLS) | `model_lr.pkl` |
| KNN Regressor | `sklearn.neighbors.KNeighborsRegressor` | `n_neighbors=5` | `model_knn.pkl` |

Evaluated using **MAE (Mean Absolute Error)** and **R² score** on the held-out test set.

The `.pkl` files are loaded into memory at Flask server startup, allowing fast predictions without retraining on every request.

---

## Inference & Grading

1. User submits 11 input fields via the web form
2. Inputs are preprocessed (encoded + scaled) to match the training pipeline
3. Both models predict independently → scores are **averaged**
4. The average is **clamped** to the range `[0, 100]`
5. The final score is converted to a **letter grade**:

| Score | Grade |
|---|---|
| ≥ 85 | A |
| ≥ 70 | B |
| ≥ 55 | C |
| ≥ 40 | D |
| ≥ 25 | E |
| < 25 | F |

---

## API Endpoints

**Base URL:** `http://localhost:5001`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Predict exam score and grade |
| `GET` | `/health` | Health check |

### Example Request

```json
POST /predict
{
  "age": 20,
  "gender": "male",
  "course": "b.sc",
  "study_hours": 6,
  "class_attendance": 85,
  "internet_access": "yes",
  "sleep_hours": 7,
  "sleep_quality": "good",
  "study_method": "online videos",
  "facility_rating": "medium",
  "exam_difficulty": "moderate"
}
```

### Example Response

```json
{
  "success": true,
  "predictions": {
    "linear_regression": 78.42,
    "knn": 74.60,
    "average": 76.51
  },
  "grade": "B",
  "timing": {
    "linear_ms": 0.6,
    "knn_ms": 18.3,
    "total_ms": 19.1
  }
}
```

---

## Web Interface

This service is connected to a **React frontend** via an **Express.js proxy server**. Users can visit the Student Performance Predictor page on the website, fill in an 11-field form (dropdowns and number inputs), and receive a live prediction showing:

- Predicted score from Linear Regression
- Predicted score from KNN
- Averaged score across both models
- Letter grade (A–F) based on the average
- Per-model inference time in milliseconds
- A live log panel showing each step of the request

---

## Files in This Repository

```
Student-Grade-Performance-Predictor/
|
|-- project_code_initial.ipynb  <- Original collaborative Jupyter notebook (group work, CSE422)
|-- train_and_save.py           <- Solo work: retraining pipeline for the Flask deployment
|-- app.py                      <- Solo work: Flask REST API server
|-- requirements.txt            <- Python dependencies
|-- .python-version             <- Python version pin
|-- README.md                   <- This file
|
|   -- Generated by train_and_save.py (NOT committed -- too large for GitHub) --
|-- scaler.pkl                  <- Fitted StandardScaler
|-- model_lr.pkl                <- Linear Regression model
|-- model_knn.pkl               <- KNN Regressor model      (~2.7 MB)
|-- model_columns.pkl           <- Ordered list of training feature columns
```

**Commit to GitHub:** `project_code_initial.ipynb`, `train_and_save.py`, `app.py`, `requirements.txt`, `.python-version`, `README.md`

**Do NOT commit:** `.pkl` files and `Exam_Score_Prediction.csv` — add them to `.gitignore`

---

## Recommended `.gitignore` Entries

```
*.pkl
Exam_Score_Prediction.csv
__pycache__/
*.pyc
.env
```

---

## Local Setup

```bash
# 1. Clone the repo and enter this directory
cd Student-Grade-Performance-Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download Exam_Score_Prediction.csv from Kaggle and place it in this directory
#    https://www.kaggle.com/datasets/kundanbedmutha/exam-score-prediction-dataset

# 4. Train the models and generate .pkl files
python train_and_save.py

# 5. Start the Flask API server
python app.py
# Runs on http://localhost:5001
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn 1.7.1 |
| Data processing | pandas 2.2.3, NumPy 2.2.6 |
| API | Flask 3.0.0, flask-cors 4.0.0 |
| Production server | gunicorn 23.0.0 |
| Frontend | React (separate repo) |
| Proxy | Express.js (separate repo) |
