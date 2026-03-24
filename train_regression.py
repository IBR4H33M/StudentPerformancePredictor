import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
csv_path = 'Exam_Score_Prediction.csv'
print(f"Loading {csv_path}...")
fr = pd.read_csv(csv_path)
print(f"Dataset rows: {len(fr)}, columns: {len(fr.columns)}")

# Drop ID and target
X = fr.drop(columns=['student_id', 'exam_score'])
y = fr['exam_score']

# Declare numeric/categorical features
numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features),
    ],
    remainder='drop'
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Build models
models = {
    'knn': KNeighborsRegressor(n_neighbors=7),
    'linear': LinearRegression(),
    'random_forest': RandomForestRegressor(n_estimators=250, random_state=42)
}

# Fit preprocessing once and transform
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# Train and evaluate
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_trans, y_train)
    y_pred_train = model.predict(X_train_trans)
    y_pred_test = model.predict(X_test_trans)
    train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    test_r2 = r2_score(y_test, y_pred_test)
    results[name] = {'model': model, 'train_rmse': train_rmse, 'test_rmse': test_rmse, 'test_r2': test_r2}
    print(f"{name}: train_rmse={train_rmse:.3f}, test_rmse={test_rmse:.3f}, test_r2={test_r2:.3f}")

# Save transformer and models
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

for name, info in results.items():
    model_file = f"model_{name}_reg.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(info['model'], f)
    print(f"Saved {model_file}")

print('Training complete')
