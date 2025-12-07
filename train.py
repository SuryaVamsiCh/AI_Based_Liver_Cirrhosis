import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# --- New Imports ---
from imblearn.pipeline import Pipeline  # Use pipeline from imblearn!
from imblearn.over_sampling import SMOTE
# --------------------
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score # Added more metrics
import joblib
import numpy as np

# --- Configuration (Keep as before) ---
DATA_FILE = 'HealthCareData.csv'
TARGET_COLUMN = 'Predicted_ValueOut_ComePatient_suffering_from_liver_cirrosis_or_not'
important_features = [
    'Duration_of_alcohol_consumptionyears', 'Total_Bilirubin_mgdl',
    'RBC_million_cellsmicroliter', 'USG_Abdomen_diffuse_liver_or_not',
    'MCHC_gramsdeciliter', 'Direct_mgdl', 'ALPhosphatase_UL',
    'Platelet_Count_lakhsmm', 'Lymphocytes_', 'AG_Ratio', 'SGOTAST_UL',
    'PCV_', 'Total_Count', 'Albumin_gdl', 'Indirect_mgdl'
]
numerical_features = [
    'Duration_of_alcohol_consumptionyears', 'Total_Bilirubin_mgdl',
    'RBC_million_cellsmicroliter', 'MCHC_gramsdeciliter', 'Direct_mgdl',
    'ALPhosphatase_UL', 'Platelet_Count_lakhsmm', 'Lymphocytes_', 'AG_Ratio',
    'SGOTAST_UL', 'PCV_', 'Total_Count', 'Albumin_gdl', 'Indirect_mgdl'
]
categorical_features = [
    'USG_Abdomen_diffuse_liver_or_not'
]
assert set(important_features) == set(numerical_features + categorical_features), "Mismatch in feature categorization!"

# --- Load Data (Keep as before) ---
try:
    data = pd.read_csv(DATA_FILE)
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: {DATA_FILE} not found.")
    exit()

data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors='coerce')
data.dropna(subset=[TARGET_COLUMN], inplace=True)
data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

X = data[important_features]
y = data[TARGET_COLUMN]

# --- Check for Class Imbalance (Keep as before) ---
print("\n--- Target Class Distribution ---")
class_counts = y.value_counts()
print(class_counts)
is_imbalanced = (class_counts.min() / class_counts.max()) < 0.1 # Keep threshold low due to severity
if is_imbalanced:
    print("⚠️ Warning: Target classes are SEVERELY imbalanced.")
else:
    print("ℹ️ Target classes appear reasonably balanced.")

# --- Split Data (Keep as before) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nData split: {len(X_train)} training samples, {len(X_test)} testing samples.")

# --- Preprocessing Steps (Keep as before) ---
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    remainder='passthrough'
)

# --- Model Definition (Keep random_state, remove class_weight as SMOTE handles balance) ---
rf_model = RandomForestClassifier(random_state=42) # Removed class_weight='balanced'

# --- Hyperparameter Tuning (GridSearchCV) ---
# Keep param_grid as before (with 'classifier__' prefix)
param_grid = {
    'classifier__n_estimators': [100, 200, 300], # Expanded slightly
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10], # Expanded slightly
    'classifier__min_samples_leaf': [1, 3, 5]    # Expanded slightly
}

# --- Create the full pipeline WITH SMOTE ---
# IMPORTANT: Use Pipeline from imblearn, not sklearn
full_pipeline_with_smote = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)), # Add SMOTE step
    ('classifier', rf_model)
])
# ---------------------------------------------

print("\n--- Starting Hyperparameter Tuning with SMOTE (GridSearchCV) ---")
# Scoring: Keep using 'f1' or 'roc_auc' due to imbalance. 'roc_auc' is often preferred.
scoring_metric = 'roc_auc' # Changed to AUC
grid_search = GridSearchCV(full_pipeline_with_smote, param_grid, cv=5, scoring=scoring_metric, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("\n--- Tuning Complete ---")
print(f"Best Score ({scoring_metric}): {grid_search.best_score_:.4f}")
print("Best Parameters:")
print(grid_search.best_params_)

# --- Evaluate Best Model ---
best_model_pipeline = grid_search.best_estimator_
print("\n--- Evaluating Best Model on Test Set ---")
y_pred = best_model_pipeline.predict(X_test)
y_pred_proba = best_model_pipeline.predict_proba(X_test)[:, 1] # Probabilities for AUC

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0) # Calculate F1 score
roc_auc = roc_auc_score(y_test, y_pred_proba) # Calculate ROC AUC

print(f"Accuracy: {accuracy:.4f}" )
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}") # AUC is a good overall metric for imbalance

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --- Save the Model ---
joblib.dump(best_model_pipeline, 'best_rf_smote_pipeline.pkl') # Changed filename
print("\n✅ Best model pipeline (with SMOTE) saved as 'best_rf_smote_pipeline.pkl'.")