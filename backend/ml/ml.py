import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

# ── 1. Create or Load Dataset ───────────────────────────────────────────────
# If you have your original CSV, replace this path:
DATASET_PATH = "data/loan_data.csv"

def get_data():
    """Loads existing data or generates a realistic synthetic dataset for testing."""
    if os.path.exists(DATASET_PATH):
        print(f"Loading dataset from {DATASET_PATH}...")
        return pd.read_csv(DATASET_PATH)
    
    print(f"Dataset not found at {DATASET_PATH}. Generating realistic synthetic data...")
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'married': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.randint(0, 4, n_samples),
        'education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'employment_status': np.random.choice(['Salaried', 'Self-employed', 'Unemployed'], n_samples),
        'applicant_income': np.random.normal(5000, 2000, n_samples).clip(1000, 15000),
        'coapplicant_income': np.random.normal(2000, 1000, n_samples).clip(0, 8000),
        'loan_amount': np.random.normal(20000, 10000, n_samples).clip(5000, 100000),
        'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'credit_score': np.random.normal(680, 80, n_samples).clip(300, 850).astype(int),
        'property_area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples),
        'existing_loans': np.random.randint(0, 3, n_samples),
        'savings': np.random.normal(10000, 5000, n_samples).clip(0, 50000),
        'collateral_value': np.random.normal(30000, 15000, n_samples).clip(0, 100000)
    })
    
    # Create target variable based on logic
    # Higher credit score, higher income, lower loan amount = more likely to be approved
    score = (data['credit_score'] / 850) * 0.4 + \
            (data['applicant_income'] / 15000) * 0.3 - \
            (data['loan_amount'] / 100000) * 0.3
            
    # Add random noise
    score += np.random.normal(0, 0.1, n_samples)
    
    # Threshold for approval (1 = Approved, 0 = Rejected)
    data['loan_status'] = (score > np.median(score)).astype(int)
    
    return data

# ── 2. Preprocessing & Training Setup ───────────────────────────────────────
df = get_data()

X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify column types
numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("\n--- Features ---")
print(f"Numeric: {numeric_features}")
print(f"Categorical: {categorical_features}")

# Build preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# ── 3. Train and Evaluate Multiple Models ───────────────────────────────────

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

print("\n--- Model Training & Evaluation ---")
best_model_name = None
best_accuracy = 0
best_pipeline = None

for name, model in models.items():
    # Create pipeline with preprocessor and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel: {name}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    
    # Track the best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_pipeline = pipeline

print("========================================")
print(f"*** Best Model: {best_model_name} with Accuracy: {best_accuracy * 100:.2f}% ***")
print("========================================")

# ── 4. Save the Best Model (Optional) ───────────────────────────────────────
# Uncomment the code below if you want to save this best pipeline to replace the current one.
# It saves the entire pipeline (scaler + encoders + model) in one .pkl file!

"""
import os
save_dir = "ML model"
os.makedirs(save_dir, exist_ok=True)
joblib.dump(best_pipeline, os.path.join(save_dir, "best_loan_pipeline.pkl"))
print(f"✅ Saved best model to {save_dir}/best_loan_pipeline.pkl")
"""
