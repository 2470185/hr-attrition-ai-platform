"""
Phase 3 — Combined Flask API
Endpoints:
  POST /predict          → attrition risk for one employee
  POST /ask              → natural language HR question to AI agent
  GET  /department-stats → attrition stats by department (for Power BI)
  GET  /high-risk        → top 10 high-risk employees
  GET  /health           → health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import pandas as pd
import joblib
import os
from phase3_ai_agent import ask_hr_agent, predict_employee_risk, run_sql

app = Flask(__name__)
CORS(app)  # Allow Power BI / frontend to call this API

# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "HR Attrition API running"})

# ─────────────────────────────────────────────
# 1. Predict attrition risk for one employee
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Body: JSON dict of employee features
    Returns: { attrition_probability, risk_level, recommendation }
    """
    try:
        data   = request.get_json()
        result = predict_employee_risk(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ─────────────────────────────────────────────
# 2. AI Agent — natural language HR questions
# ─────────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask():
    """
    Body: { "question": "Which department has highest attrition?" }
    Returns: { question, sql, answer, rows }
    """
    try:
        body     = request.get_json()
        question = body.get("question", "").strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400

        result = ask_hr_agent(question, verbose=False)

        return jsonify({
            "question": result["question"],
            "sql":      result["sql"],
            "answer":   result["answer"],
            "rows":     result["data"].to_dict(orient="records") if result["data"] is not None else []
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# 3. Department attrition stats (for Power BI)
# ─────────────────────────────────────────────
@app.route("/department-stats", methods=["GET"])
def department_stats():
    """
    Returns attrition count and rate per department.
    Connect Power BI to this endpoint via Web connector.
    """
    try:
        sql = """
        SELECT
            "Department",
            COUNT(*) AS total_employees,
            SUM(CASE WHEN "Attrition" = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
            ROUND(
                100.0 * SUM(CASE WHEN "Attrition" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
                2
            ) AS attrition_rate_pct,
            ROUND(AVG("MonthlyIncome"), 2) AS avg_monthly_income,
            ROUND(AVG("YearsAtCompany"), 2) AS avg_tenure_years
        FROM employees
        GROUP BY "Department"
        ORDER BY attrition_rate_pct DESC
        """
        df = run_sql(sql)
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# 4. Top high-risk employees
# ─────────────────────────────────────────────
@app.route("/high-risk", methods=["GET"])
def high_risk():
    """
    Returns top 10 employees by predicted attrition risk.
    """
    try:
        sql = """
        SELECT
            "EmployeeNumber",
            "Age",
            "Department",
            "JobRole",
            "MonthlyIncome",
            "YearsAtCompany",
            "OverTime",
            "attrition_risk"
        FROM employees
        WHERE "Attrition" = 'No'   -- show only current employees at risk
        ORDER BY "attrition_risk" DESC
        LIMIT 10
        """
        df = run_sql(sql)
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# 5. Overall KPI summary (for Power BI cards)
# ─────────────────────────────────────────────
@app.route("/kpi-summary", methods=["GET"])
def kpi_summary():
    try:
        sql = """
        SELECT
            COUNT(*) AS total_employees,
            SUM(CASE WHEN "Attrition" = 'Yes' THEN 1 ELSE 0 END) AS total_attritions,
            ROUND(100.0 * SUM(CASE WHEN "Attrition" = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS overall_attrition_rate,
            ROUND(AVG("MonthlyIncome"), 2) AS avg_monthly_income,
            ROUND(AVG("YearsAtCompany"), 2) AS avg_tenure,
            SUM(CASE WHEN "attrition_risk" > 0.5 THEN 1 ELSE 0 END) AS high_risk_count
        FROM employees
        """
        df = run_sql(sql)
        return jsonify(df.to_dict(orient="records")[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting HR Attrition API on http://localhost:5000")
    print("Endpoints:")
    print("  GET  /health")
    print("  POST /predict")
    print("  POST /ask")
    print("  GET  /department-stats")
    print("  GET  /high-risk")
    print("  GET  /kpi-summary")
    app.run(debug=True, port=5000)
