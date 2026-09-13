"""Honest evaluation: drop the leaked column and the constant columns.

`attrition_risk` is the deployed model's own score, backfilled into the
employees table by Phase 3. Training on it leaks the target (corr 0.91) and
inflates every metric. The original Phase 2 run would have preceded that
backfill, so this clean run is the fair reconstruction of it.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

SRC = (r"C:\Users\Kuldeep Patra\OneDrive - kiit.ac.in\Documents"
       r"\AI-Powered HR Analytics & Attrition Prediction System\employees.json")

raw = json.load(open(SRC))
df = pd.DataFrame(raw if isinstance(raw, list) else list(raw.values())[0])

LEAK = ['attrition_risk']
CONST = [c for c in df.columns if df[c].nunique() <= 1]
drop = [c for c in LEAK + CONST if c in df.columns]
print(f"dropped: leaked={LEAK}  constant={CONST}")

X = df.drop(columns=['Attrition'] + drop)
y = df['Attrition']
print(f"features: {X.shape[1]}   rows: {len(X)}   attrition rate: {y.mean():.3f}\n")

# --- notebook's own split, now clean -----------------------------------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
m = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
m.fit(Xtr, ytr)
prob = m.predict_proba(Xte)[:, 1]
pred = m.predict(Xte)

print("--- notebook split (80/20, seed 42), leakage removed ---")
print(confusion_matrix(yte, pred))
print(classification_report(yte, pred, digits=3))
print(f"ROC-AUC: {roc_auc_score(yte, prob):.4f}\n")

# --- seed sensitivity ---------------------------------------------------
print("--- seed sensitivity ---")
aucs = []
for s in (0, 1, 7, 42, 2024):
    a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)
    mm = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    mm.fit(a, c)
    au = roc_auc_score(d, mm.predict_proba(b)[:, 1])
    aucs.append(au)
    print(f"  seed {s:<5} ROC-AUC {au:.4f}  recall {recall_score(d, mm.predict(b)):.3f}")
print(f"  mean {np.mean(aucs):.4f}  min {min(aucs):.4f}  max {max(aucs):.4f}\n")

# --- 5-fold stratified CV: the figure worth quoting ---------------------
print("--- 5-fold stratified CV (headline figure) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold, recs = [], []
for tr, te in skf.split(X, y):
    mm = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    mm.fit(X.iloc[tr], y.iloc[tr])
    fold.append(roc_auc_score(y.iloc[te], mm.predict_proba(X.iloc[te])[:, 1]))
    recs.append(recall_score(y.iloc[te], mm.predict(X.iloc[te])))
print(f"  ROC-AUC per fold: {', '.join(f'{f:.3f}' for f in fold)}")
print(f"  ROC-AUC  mean {np.mean(fold):.4f} +/- {np.std(fold):.4f}")
print(f"  recall   mean {np.mean(recs):.4f} +/- {np.std(recs):.4f}")

# --- top features -------------------------------------------------------
imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- top 10 features by gain ---")
for k, v in imp.head(10).items():
    print(f"  {k:<28s} {v:.4f}")
