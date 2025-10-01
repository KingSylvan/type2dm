import sqlite3
import pandas as pd
from datetime import datetime
import json

def init_db():
    """Initialize SQLite database for storing model results and predictions"""
    conn = sqlite3.connect('db/predictions.db')
    c = conn.cursor()
    
    # Create predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  age INTEGER,
                  sex TEXT,
                  bmi REAL,
                  waist REAL,
                  smoking_status TEXT,
                  alcohol_freq TEXT,
                  moderate_mins INTEGER,
                  vigorous_mins INTEGER,
                  sleep_hrs REAL,
                  fruit_intake INTEGER,
                  veg_intake INTEGER,
                  red_meat_intake INTEGER,
                  predicted_risk REAL,
                  risk_category TEXT,
                  lifestyle_score INTEGER)''')
    
    # Create model_results table
    c.execute('''CREATE TABLE IF NOT EXISTS model_results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  model_name TEXT,
                  auc_score REAL,
                  f1_score REAL,
                  precision_score REAL,
                  recall_score REAL,
                  feature_importance TEXT,
                  model_params TEXT)''')
    
    # Create feature_importance table
    c.execute('''CREATE TABLE IF NOT EXISTS feature_importance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  model_id INTEGER,
                  feature_name TEXT,
                  importance_score REAL,
                  FOREIGN KEY(model_id) REFERENCES model_results(id))''')
    
    conn.commit()
    conn.close()

def save_prediction(user_data, risk_prob, risk_category, lifestyle_score):
    """Save prediction to database"""
    conn = sqlite3.connect('db/predictions.db')
    c = conn.cursor()
    
    c.execute('''INSERT INTO predictions 
                 (timestamp, age, sex, bmi, waist, smoking_status, alcohol_freq,
                  moderate_mins, vigorous_mins, sleep_hrs, fruit_intake, veg_intake,
                  red_meat_intake, predicted_risk, risk_category, lifestyle_score)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               user_data['age'], user_data['sex'], user_data['bmi'], user_data['waist'],
               user_data['smoking_status'], user_data['alcohol_freq'],
               user_data['moderate_mins'], user_data['vigorous_mins'], user_data['sleep_hrs'],
               user_data['fruit_intake'], user_data['veg_intake'], user_data['red_meat_intake'],
               risk_prob, risk_category, lifestyle_score))
    
    conn.commit()
    conn.close()

def save_model_results(model_name, results, feature_importance=None, model_params=None):
    """Save model results and feature importance to database"""
    conn = sqlite3.connect('db/predictions.db')
    c = conn.cursor()
    
    # Save model results
    c.execute('''INSERT INTO model_results 
                 (timestamp, model_name, auc_score, f1_score, precision_score, recall_score,
                  feature_importance, model_params)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               model_name,
               results.get('auc', 0.0),
               results.get('f1', 0.0),
               results.get('precision', 0.0),
               results.get('recall', 0.0),
               json.dumps(feature_importance) if feature_importance else None,
               json.dumps(model_params) if model_params else None))
    
    # Get the ID of the inserted model result
    model_id = c.lastrowid
    
    # Save feature importance if provided
    if feature_importance:
        for feature, importance in feature_importance.items():
            c.execute('''INSERT INTO feature_importance 
                        (model_id, feature_name, importance_score)
                        VALUES (?, ?, ?)''',
                     (model_id, feature, float(importance)))
    
    conn.commit()
    conn.close()

def fetch_recent_predictions(limit=10):
    """Fetch recent predictions from database"""
    conn = sqlite3.connect('db/predictions.db')
    df = pd.read_sql_query(f"SELECT * FROM predictions ORDER BY timestamp DESC LIMIT {limit}", conn)
    conn.close()
    return df

def fetch_model_results(model_name=None):
    """Fetch model results from database"""
    conn = sqlite3.connect('db/predictions.db')
    
    if model_name:
        query = "SELECT * FROM model_results WHERE model_name = ? ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn, params=(model_name,))
    else:
        df = pd.read_sql_query("SELECT * FROM model_results ORDER BY timestamp DESC", conn)
    
    conn.close()
    return df

def fetch_feature_importance(model_id):
    """Fetch feature importance for a specific model"""
    conn = sqlite3.connect('db/predictions.db')
    df = pd.read_sql_query("""
        SELECT feature_name, importance_score 
        FROM feature_importance 
        WHERE model_id = ? 
        ORDER BY importance_score DESC""", 
        conn, params=(model_id,))
    conn.close()
    return df

def get_model_performance_summary():
    """Get summary of model performance over time"""
    conn = sqlite3.connect('db/predictions.db')
    df = pd.read_sql_query("""
        SELECT model_name,
               AVG(auc_score) as avg_auc,
               AVG(f1_score) as avg_f1,
               AVG(precision_score) as avg_precision,
               AVG(recall_score) as avg_recall,
               COUNT(*) as num_runs
        FROM model_results
        GROUP BY model_name""", conn)
    conn.close()
    return df