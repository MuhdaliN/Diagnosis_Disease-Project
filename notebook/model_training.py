# %% [markdown]
# # Disease Prediction – Advanced Ensemble with SMOTE & Hyperparameter Tuning

# %%
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, f1_score, make_scorer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils.class_weight import compute_class_weight


# %% [markdown]
# ### 1. Load and Clean Data

# %%
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

df = pd.read_csv(PROJECT_ROOT / 'Training.csv')


# Fill missing values
df.fillna(0, inplace=True)

# Clean column names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
# Fix specific problematic names
df.rename(columns={'spotting__urination':'spotting_urination', 
                   'foul_smell_of_urine':'foul_smell_urine',
                   'toxic_look_(typhos)':'toxic_look_typhos',
                   'dischromic__patches':'dischromic_patches'}, inplace=True)

# Target encoding
le = LabelEncoder()
df['prognosis'] = le.fit_transform(df['prognosis'])

# Features and target
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# Save symptom names and label encoder for later use in Django
symptom_columns = X.columns.tolist()
joblib.dump(symptom_columns, 'symptom_columns.pkl')
joblib.dump(le, 'label_encoder.pkl')

# %% [markdown]
# ### 2. Feature Engineering & Selection

# %%
# ----- Interaction Terms (Domain‑aware) -----
# We create pairwise products for the top 15 most frequent symptoms.
# This captures co‑occurrence patterns (medical correlations).
from collections import Counter

symptom_freq = X.sum().sort_values(ascending=False)
top_symptoms = symptom_freq.head(15).index.tolist()

for i, sym1 in enumerate(top_symptoms):
    for sym2 in top_symptoms[i+1:]:
        X[f'{sym1}_&_{sym2}'] = X[sym1] * X[sym2]

print(f"Added {len(top_symptoms)*(len(top_symptoms)-1)//2} interaction terms.")

# ----- Feature Selection (Mutual Information) -----
selector = SelectKBest(mutual_info_classif, k=min(200, X.shape[1]))  # keep top 200 features
X_selected = selector.fit_transform(X, y)
selected_indices = selector.get_support(indices=True)
selected_features = [X.columns[i] for i in selected_indices]

# Save selected features for Django
joblib.dump(selected_features, 'selected_features.pkl')

# Reduce X to selected features
X = X[selected_features]

# %% [markdown]
# ### 3. Train/Test Split & SMOTE

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:", Counter(y_train_res))

# %% [markdown]
# ### 4. Ensemble Model & Hyperparameter Optimization

# %%
# Base models
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
xgb_clf = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

# Voting classifier (soft voting)
ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('xgb', xgb_clf),
    ('lr', lr)
], voting='soft')

# Custom weighted log loss for medical context (penalize rare diseases more)
def weighted_log_loss(y_true, y_pred_proba, class_weights=None):
    if class_weights is None:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_true), y=y_true)
        class_weights = dict(zip(np.unique(y_true), class_weights))
    sample_weights = np.array([class_weights[y] for y in y_true])
    return log_loss(y_true, y_pred_proba, sample_weight=sample_weights)

# Hyperparameter grid (simplified for time, expandable)
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 20],
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 6],
    'lr__C': [0.1, 1.0]
}

# Use randomized search with weighted log loss as scoring
scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
random_search = RandomizedSearchCV(
    ensemble, param_grid, n_iter=5, cv=3, scoring=scorer, random_state=42, n_jobs=-1
)
random_search.fit(X_train_res, y_train_res)

best_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)

# %% [markdown]
# ### 5. Evaluation with Confidence Intervals

# %%
from sklearn.utils import resample

# Bootstrap for confidence intervals
n_iterations = 100
test_scores = []
for i in range(n_iterations):
    X_bs, y_bs = resample(X_test, y_test, replace=True, random_state=i)
    y_pred_proba_bs = best_model.predict_proba(X_bs)
    loss = weighted_log_loss(y_bs, y_pred_proba_bs)
    test_scores.append(loss)

lower = np.percentile(test_scores, 2.5)
upper = np.percentile(test_scores, 97.5)
print(f"Weighted log loss: {np.mean(test_scores):.4f} (95%% CI: [{lower:.4f}, {upper:.4f}])")

# Also compute F1-score per class (subgroup analysis)
from sklearn.metrics import classification_report
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# %% [markdown]
# ### 6. Model Interpretability (SHAP)

# %%
import numpy as np
np.bool = bool  # Fix for SHAP compatibility

import shap

explainer = shap.TreeExplainer(best_model.named_estimators_['rf'])
shap_values = explainer.shap_values(X_test[:50])
shap.summary_plot(shap_values, X_test[:50], feature_names=selected_features)


# %% [markdown]
# ### 7. Save Model

# %%
joblib.dump(best_model, 'disease_prediction_ensemble.pkl')
print("Model saved.")