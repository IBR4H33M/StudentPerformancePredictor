import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, r2_score

# ── Load dataset ──────────────────────────────────────────────
df = pd.read_csv('Exam_Score_Prediction.csv')

# Drop student_id — not a feature
df.drop(columns=['student_id'], inplace=True)

# ── Encoding ──────────────────────────────────────────────────

# Label encoding
df['gender']          = df['gender'].map({'male': 0, 'female': 1, 'other': 2})
df['internet_access'] = df['internet_access'].map({'no': 0, 'yes': 1})
df['sleep_quality']   = df['sleep_quality'].map({'poor': 0, 'average': 1, 'good': 2})
df['facility_rating'] = df['facility_rating'].map({'low': 0, 'medium': 1, 'high': 2})
df['exam_difficulty'] = df['exam_difficulty'].map({'easy': 0, 'moderate': 1, 'hard': 2})

# One-hot encoding
df = pd.get_dummies(df, columns=['course', 'study_method'])

# ── Split features / target ───────────────────────────────────
X = df.drop(columns=['exam_score'])
y = df['exam_score']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Scale ─────────────────────────────────────────────────────
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)

# ── Train models ──────────────────────────────────────────────
model_lr  = LinearRegression()
model_knn = KNeighborsRegressor(n_neighbors=5)

model_lr.fit(x_train_scaled, y_train)
model_knn.fit(x_train_scaled, y_train)

# ── Evaluate ──────────────────────────────────────────────────
for name, model in [('Linear Regression', model_lr), ('KNN Regressor', model_knn)]:
    preds = model.predict(x_test_scaled)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f'{name} — MAE: {mae:.2f}, R²: {r2:.4f}')

# ── Save ──────────────────────────────────────────────────────
pickle.dump(scaler,       open('scaler.pkl', 'wb'))
pickle.dump(model_lr,     open('model_lr.pkl', 'wb'))
pickle.dump(model_knn,    open('model_knn.pkl', 'wb'))
pickle.dump(list(X.columns), open('model_columns.pkl', 'wb'))

print("\n✅ Models saved: scaler.pkl, model_lr.pkl, model_knn.pkl, model_columns.pkl")
