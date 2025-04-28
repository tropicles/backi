import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.pipeline import Pipeline
import joblib
from collections import Counter

# 1. Load data (skip any malformed lines)
df = pd.read_csv(
    "sd.csv"
)

# 2. One-hot encode 'type'
ohe = OneHotEncoder(drop='first', sparse_output=False)
type_encoded = ohe.fit_transform(df[['type']])
type_cols = ohe.get_feature_names_out(['type'])
type_df = pd.DataFrame(type_encoded, columns=type_cols, index=df.index)

# 3. Feature engineering
df['balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']

# 4. Assemble X and y
X = pd.concat([
    df[['amount',
        'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest',
        'balance_diff']],
    type_df
], axis=1)
y = df['isFraud']

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Set SMOTE k_neighbors safely
minority_count = Counter(y_train)[1]
if minority_count > 1:
    k = min(5, minority_count - 1)
    sampler = SMOTE(random_state=42, k_neighbors=k)
else:
    sampler = RandomOverSampler(random_state=42)

# 7. Resample
X_res, y_res = sampler.fit_resample(X_train, y_train)

# 8. Build & fit pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ))
])
pipeline.fit(X_res, y_res)

# 9. Evaluate
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:",      accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# 10. Confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True, fmt='d', cmap='Blues',
    xticklabels=['Not Fraud','Fraud'],
    yticklabels=['Not Fraud','Fraud']
)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# 11. Proper S-curve for 'amount'

# a) Extract scaler & model, locate 'amount' index
scaler       = pipeline.named_steps['scaler']
model        = pipeline.named_steps['model']
amount_idx   = list(X_train.columns).index('amount')
mean_amt     = scaler.mean_[amount_idx]
scale_amt    = scaler.scale_[amount_idx]
coef_amt     = model.coef_[0][amount_idx]
intercept    = model.intercept_[0]

# b) Choose a focused grid in raw 'amount' space (1st–99th percentiles)
p1, p99 = np.percentile(X_train['amount'], [1, 99])
grid_raw = np.linspace(p1, p99, 300)

# c) Compute logits and probabilities
grid_scaled = (grid_raw - mean_amt) / scale_amt
z           = intercept + coef_amt * grid_scaled
p_curve     = 1 / (1 + np.exp(-z))

# d) Plot
plt.figure(figsize=(8,4))
# scatter actual test points
plt.scatter(
    X_test['amount'],
    y_test + np.random.uniform(-0.02, 0.02, size=y_test.shape),
    alpha=0.3,
    label='Actual (jittered)'
)
# true S-curve
plt.plot(
    grid_raw,
    p_curve,
    linewidth=2,
    label='Sigmoid P(fraud) vs. amount'
)
plt.xlabel('Transaction Amount')
plt.ylabel('Fraud Label / Predicted Probability')
plt.title('Logistic Regression S-Curve on Amount\n(others fixed at mean)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 12. Save artifacts
joblib.dump(pipeline, "fraud_detector.pkl")
print("Model saved as fraud_detector.pkl")
joblib.dump(ohe, "type_encoder.pkl")
print("Encoder saved as type_encoder.pkl")
