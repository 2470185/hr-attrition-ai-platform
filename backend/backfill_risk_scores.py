"""
backfill_risk_scores.py
Run ONCE after Phase 2 to write predicted attrition_risk scores
back into your Neon PostgreSQL employees table.
"""

import psycopg2
import pandas as pd
import joblib
from phase3_ai_agent import NEON_CONN_STR, MODEL_PATH

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Fetching employees from Neon...")
conn = psycopg2.connect(NEON_CONN_STR)
df   = pd.read_sql_query('SELECT * FROM employees', conn)
conn.close()

print(f"Loaded {len(df)} employees")

# ── Encode categoricals the same way as training ──────────────
from sklearn.preprocessing import LabelEncoder

cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c not in ['Attrition', 'EmployeeNumber']]

le = LabelEncoder()
df_enc = df.copy()
for col in cat_cols:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

# ── Select only the features the model was trained on ─────────
if hasattr(model, 'feature_names_in_'):
    feature_cols = list(model.feature_names_in_)
else:
    # fallback: drop non-feature columns
    drop_cols = ['Attrition', 'EmployeeNumber', 'attrition_risk',
                 'EmployeeCount', 'Over18', 'StandardHours']
    feature_cols = [c for c in df_enc.columns if c not in drop_cols]

X    = df_enc[feature_cols]
prob = model.predict_proba(X)[:, 1]

print(f"Predicted risks: min={prob.min():.3f}  max={prob.max():.3f}  mean={prob.mean():.3f}")

# ── Write risk scores back to Neon ────────────────────────────
print("Writing risk scores to Neon...")
conn   = psycopg2.connect(NEON_CONN_STR)
cursor = conn.cursor()

# Make sure column exists
cursor.execute("ALTER TABLE employees ADD COLUMN IF NOT EXISTS attrition_risk FLOAT DEFAULT 0.0;")

updated = 0
for idx, row in df.iterrows():
    risk = float(prob[idx])
    cursor.execute(
        'UPDATE employees SET attrition_risk = %s WHERE "EmployeeNumber" = %s',
        (risk, row['EmployeeNumber'])
    )
    updated += 1

conn.commit()
cursor.close()
conn.close()

print(f"Done! Updated attrition_risk for {updated} employees in Neon.")
print("You can now run: python app.py")
