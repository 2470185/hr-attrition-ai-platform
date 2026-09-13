"""
Phase 3 — AI Agent Layer
HR Attrition Prediction System
Uses: Groq (free LLaMA 3.1 70B) + Neon PostgreSQL + trained model
"""

import os
import psycopg2
import pandas as pd
import joblib
import json
from groq import Groq

# ─────────────────────────────────────────────
# CONFIG — fill these in
# ─────────────────────────────────────────────
GROQ_API_KEY   = "your_groq_api_key_here"       # console.groq.com → free
NEON_CONN_STR  = "postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require"
MODEL_PATH     = "model.pkl"                     # saved from Phase 2
# ─────────────────────────────────────────────

client = Groq(api_key=GROQ_API_KEY)

# ── DB helper ──────────────────────────────────
def run_sql(query: str) -> pd.DataFrame:
    """Run a SELECT query on Neon and return a DataFrame."""
    conn = psycopg2.connect(NEON_CONN_STR)
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    return df

# ── Schema description sent to LLM ─────────────
TABLE_SCHEMA = """
Table name: employees
Columns:
  - "Age" INTEGER
  - "Attrition" VARCHAR  (values: 'Yes' or 'No')
  - "Department" VARCHAR  (e.g. 'Sales', 'Research & Development', 'Human Resources')
  - "JobRole" VARCHAR
  - "MonthlyIncome" INTEGER
  - "OverTime" VARCHAR  (values: 'Yes' or 'No')
  - "YearsAtCompany" INTEGER
  - "JobSatisfaction" INTEGER  (1=Low, 2=Medium, 3=High, 4=Very High)
  - "WorkLifeBalance" INTEGER  (1=Bad, 2=Good, 3=Better, 4=Best)
  - "EnvironmentSatisfaction" INTEGER
  - "YearsSinceLastPromotion" INTEGER
  - "DistanceFromHome" INTEGER
  - "Gender" VARCHAR
  - "MaritalStatus" VARCHAR
  - "NumCompaniesWorked" INTEGER
  - "TotalWorkingYears" INTEGER
  - "attrition_risk" FLOAT  (predicted risk 0-1, added in Phase 2)
"""

# ── Step 1: LLM generates SQL from question ───
def question_to_sql(user_question: str) -> str:
    system_prompt = f"""You are an expert PostgreSQL data analyst.
Convert the user's HR question into a valid PostgreSQL SELECT query.
Use the table schema below. Return ONLY the SQL query, nothing else.
No markdown, no explanation, no backticks.

{TABLE_SCHEMA}

Rules:
- Always use double quotes around column names with mixed case e.g. "Department"
- For attrition counts use: WHERE "Attrition" = 'Yes'
- For risk queries use: ORDER BY "attrition_risk" DESC
- Limit results to 10 rows unless asked otherwise
- Never use INSERT, UPDATE, DELETE, DROP
"""
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_question}
        ],
        temperature=0.1,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# ── Step 2: Run SQL, get results ───────────────
def execute_query(sql: str) -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, error_message). One will be None."""
    try:
        df = run_sql(sql)
        return df, None
    except Exception as e:
        return None, str(e)

# ── Step 3: LLM explains results in plain English ──
def results_to_answer(user_question: str, sql: str, df: pd.DataFrame) -> str:
    data_preview = df.to_string(index=False) if len(df) <= 20 else df.head(10).to_string(index=False)

    system_prompt = """You are an expert HR analytics consultant.
You will be given:
1. The user's original question
2. The SQL that was run
3. The query results as a table

Give a clear, insightful answer in 2-4 sentences.
Highlight the key number or finding. Add one actionable HR recommendation.
Be concise and professional."""

    user_msg = f"""Question: {user_question}

SQL used:
{sql}

Query results:
{data_preview}
"""
    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0.3,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()

# ── Main agent function ────────────────────────
def ask_hr_agent(question: str, verbose: bool = True) -> dict:
    """
    Full pipeline: question → SQL → DB → LLM answer
    Returns dict with keys: question, sql, data, answer
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    # Step 1: Generate SQL
    sql = question_to_sql(question)
    if verbose:
        print(f"\nGenerated SQL:\n{sql}")

    # Step 2: Execute SQL
    df, error = execute_query(sql)
    if error:
        print(f"SQL Error: {error}")
        # Try a fallback safe query
        sql = 'SELECT "Department", COUNT(*) as total FROM employees GROUP BY "Department"'
        df, error = execute_query(sql)
        if error:
            return {"question": question, "sql": sql, "data": None, "answer": f"Error: {error}"}

    if verbose:
        print(f"\nQuery Results ({len(df)} rows):")
        print(df.to_string(index=False))

    # Step 3: Generate natural language answer
    answer = results_to_answer(question, sql, df)
    if verbose:
        print(f"\nAgent Answer:\n{answer}")

    return {
        "question": question,
        "sql":      sql,
        "data":     df,
        "answer":   answer
    }

# ── Predict risk for a single employee ────────
def predict_employee_risk(employee_data: dict) -> dict:
    """
    Predict attrition risk for one employee using the saved model.
    employee_data: dict of feature values matching training features.
    """
    model = joblib.load(MODEL_PATH)
    df    = pd.DataFrame([employee_data])

    # Ensure columns match training data (model stores feature names)
    if hasattr(model, 'feature_names_in_'):
        df = df[model.feature_names_in_]

    prob  = model.predict_proba(df)[0][1]
    label = "HIGH RISK" if prob > 0.5 else "LOW RISK"

    return {
        "attrition_probability": round(float(prob), 3),
        "risk_level":            label,
        "recommendation":        "Schedule retention interview" if prob > 0.5 else "Employee stable"
    }

# ─────────────────────────────────────────────────
# 3 DEMO QUESTIONS — run these for your presentation
# ─────────────────────────────────────────────────
DEMO_QUESTIONS = [
    "Which department has the highest attrition rate?",
    "Show me the top 5 employees with the highest predicted attrition risk",
    "What is the average monthly income of employees who left vs those who stayed?",
]

if __name__ == "__main__":
    print("HR Attrition AI Agent — Phase 3 Demo")
    print("Powered by: Groq LLaMA 3.1 70B + Neon PostgreSQL\n")

    for q in DEMO_QUESTIONS:
        result = ask_hr_agent(q)
        print("\n" + "-"*60)

    # Example: predict risk for one employee
    sample_employee = {
        "Age": 32,
        "MonthlyIncome": 3500,
        "OverTime": 1,           # 1 = Yes (encoded)
        "YearsAtCompany": 2,
        "JobSatisfaction": 2,
        "WorkLifeBalance": 1,
        "YearsSinceLastPromotion": 3,
        "DistanceFromHome": 25,
        "TotalWorkingYears": 5,
        "NumCompaniesWorked": 3,
    }
    print("\n--- Single Employee Risk Prediction ---")
    risk = predict_employee_risk(sample_employee)
    print(json.dumps(risk, indent=2))
