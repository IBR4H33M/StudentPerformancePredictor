from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import time

app = Flask(__name__)
CORS(app)

# ── Load models ───────────────────────────────────────────────
print("Loading models...")
model_lr     = pickle.load(open('model_lr.pkl',      'rb'))
model_knn    = pickle.load(open('model_knn.pkl',     'rb'))
scaler       = pickle.load(open('scaler.pkl',        'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))
print("Models loaded successfully!")

# ── Encoding maps (must match train_and_save.py) ──────────────
LABEL_MAPS = {
    'gender':          {'male': 0, 'female': 1, 'other': 2},
    'internet_access': {'no': 0, 'yes': 1},
    'sleep_quality':   {'poor': 0, 'average': 1, 'good': 2},
    'facility_rating': {'low': 0, 'medium': 1, 'high': 2},
    'exam_difficulty': {'easy': 0, 'moderate': 1, 'hard': 2},
}

ONE_HOT_PREFIXES = ['course', 'study_method']

REQUIRED_FIELDS = [
    'age', 'gender', 'course', 'study_hours', 'class_attendance',
    'internet_access', 'sleep_hours', 'sleep_quality', 'study_method',
    'facility_rating', 'exam_difficulty'
]

def preprocess_input(data):
    df = pd.DataFrame([data])

    # Numeric fields
    for col in ['age', 'study_hours', 'class_attendance', 'sleep_hours']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    # Label encode
    for col, mapping in LABEL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].str.lower().map(mapping).fillna(0)

    # One-hot encode
    df = pd.get_dummies(df, columns=ONE_HOT_PREFIXES)

    # Add any missing columns (from training) as 0
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only training columns in correct order
    df = df[model_columns]

    return scaler.transform(df)


def clamp(value, lo=0.0, hi=100.0):
    return round(max(lo, min(hi, float(value))), 2)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Validate required fields
        missing = [f for f in REQUIRED_FIELDS if not data or f not in data or str(data.get(f, '')).strip() == '']
        if missing:
            return jsonify({'success': False, 'error': f'Missing fields: {", ".join(missing)}'}), 400

        X = preprocess_input(data)

        t0 = time.time()

        t_knn_start = time.time()
        pred_knn = clamp(model_knn.predict(X)[0])
        t_knn_ms = round((time.time() - t_knn_start) * 1000, 2)

        t_lr_start = time.time()
        pred_lr = clamp(model_lr.predict(X)[0])
        t_lr_ms = round((time.time() - t_lr_start) * 1000, 2)

        total_ms = round((time.time() - t0) * 1000, 2)

        avg = clamp((pred_lr + pred_knn) / 2)

        def to_grade(score):
            if score >= 85: return 'A'
            elif score >= 70: return 'B'
            elif score >= 55: return 'C'
            elif score >= 40: return 'D'
            elif score >= 25: return 'E'
            else: return 'F'

        return jsonify({
            'success': True,
            'predictions': {
                'linear_regression': pred_lr,
                'knn':               pred_knn,
                'average':           avg,
            },
            'timing': {
                'knn_ms': t_knn_ms,
                'linear_ms': t_lr_ms,
                'total_ms': total_ms
            },
            'grade': to_grade(avg)
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'models': 'loaded'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
