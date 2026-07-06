import os
import json
import sqlite3
import psycopg2
from pathlib import Path
from datetime import datetime


# PostgreSQL Connection Configuration
DB_CONFIG = {
    'dbname': 'power_analytics_db',
    'user': 'prince',
    'password': '9999',
    'host': 'localhost'
}

USE_SQLITE = False

def get_connection():
    """
    Get a database connection. Tries PostgreSQL first;
    falls back to SQLite on connection failure.
    """
    global USE_SQLITE
    if USE_SQLITE:
        os.makedirs('instance', exist_ok=True)
        conn = sqlite3.connect('instance/power_analytics.db')
        return conn, True
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn, False
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}. Falling back to SQLite.")
        USE_SQLITE = True
        os.makedirs('instance', exist_ok=True)
        conn = sqlite3.connect('instance/power_analytics.db')
        return conn, True

def init_db():
    """
    Initialize the database, creating users, user_history, and prediction_history
    tables in either PostgreSQL or SQLite depending on active connection.
    """
    conn, is_sqlite = get_connection()
    cur = conn.cursor()
    
    if is_sqlite:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                graph_type TEXT NOT NULL,
                filters TEXT NOT NULL,
                graph_path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                state TEXT NOT NULL,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                power_source TEXT NOT NULL,
                forecast_period TEXT NOT NULL,
                predicted_demand REAL,
                confidence REAL,
                expected_growth REAL,
                trend TEXT,
                best_model_used TEXT,
                accuracy REAL,
                metrics TEXT,
                prediction_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                graph_type VARCHAR(50) NOT NULL,
                filters JSON NOT NULL,
                graph_path VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                state VARCHAR(100) NOT NULL,
                year INTEGER NOT NULL,
                month INTEGER NOT NULL,
                power_source VARCHAR(100) NOT NULL,
                forecast_period VARCHAR(50) NOT NULL,
                predicted_demand DOUBLE PRECISION,
                confidence DOUBLE PRECISION,
                expected_growth DOUBLE PRECISION,
                trend VARCHAR(50),
                best_model_used VARCHAR(100),
                accuracy DOUBLE PRECISION,
                metrics JSON NOT NULL,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    conn.commit()
    cur.close()
    conn.close()

def load_user(user_id):
    """
    Retrieve user by ID for Flask-Login.
    """
    conn, is_sqlite = get_connection()
    cur = conn.cursor()
    param_char = '?' if is_sqlite else '%s'
    
    try:
        cur.execute(f"SELECT id, username FROM users WHERE id = {param_char}", (int(user_id),))
        user_data = cur.fetchone()
        if user_data:
            from models.user import User
            return User(user_data[0], user_data[1])
        return None
    except Exception as e:
        print(f"Error loading user: {e}")
        return None
    finally:
        cur.close()
        conn.close()

def save_user_history(user_id, graph_type, filters, graph_path=None):
    """
    Save user's visualization generation history.
    """
    conn, is_sqlite = get_connection()
    cur = conn.cursor()
    param_char = '?' if is_sqlite else '%s'
    
    if isinstance(filters, dict):
        filters_json = json.dumps(filters)
    else:
        filters_json = filters
        
    try:
        cur.execute(
            f"INSERT INTO user_history (user_id, graph_type, filters, graph_path) VALUES ({param_char}, {param_char}, {param_char}, {param_char})",
            (user_id, graph_type, filters_json, graph_path)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error saving user history: {e}")
    finally:
        cur.close()
        conn.close()

def get_user_history(user_id, limit=10):
    """
    Get recent visualization history for a user.
    """
    conn, is_sqlite = get_connection()
    cur = conn.cursor()
    param_char = '?' if is_sqlite else '%s'
    
    try:
        cur.execute(
            f"SELECT id, graph_type, filters, graph_path, created_at FROM user_history "
            f"WHERE user_id = {param_char} ORDER BY created_at DESC LIMIT {param_char}",
            (user_id, limit)
        )
        history = cur.fetchall()
        
        history_list = []
        for row in history:
            filters = row[2]
            if isinstance(filters, str):
                try:
                    filters = json.loads(filters)
                except json.JSONDecodeError:
                    filters = {}
            
            created_at = row[4]
            if isinstance(created_at, str):
                try:
                    if '.' in created_at:
                        created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S.%f')
                    else:
                        created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    created_at = datetime.now()
            
            history_list.append({
                'id': row[0],
                'graph_type': row[1],
                'filters': filters,
                'graph_path': row[3],
                'created_at': created_at
            })
        return history_list
    except Exception as e:
        print(f"Error retrieving user history: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def save_prediction_history(user_id, state, year, month, power_source, forecast_period, predicted_demand, confidence, expected_growth, trend, best_model_used, accuracy, metrics):
    """
    Save future demand predictions to prediction_history table.
    """
    conn, is_sqlite = get_connection()
    cur = conn.cursor()
    param_char = '?' if is_sqlite else '%s'
    
    if isinstance(metrics, dict):
        metrics_json = json.dumps(metrics)
    else:
        metrics_json = metrics
        
    try:
        if is_sqlite:
            cur.execute(
                f"INSERT INTO prediction_history (user_id, state, year, month, power_source, forecast_period, "
                f"predicted_demand, confidence, expected_growth, trend, best_model_used, accuracy, metrics) "
                f"VALUES ({param_char}, {param_char}, {param_char}, {param_char}, {param_char}, {param_char}, "
                f"{param_char}, {param_char}, {param_char}, {param_char}, {param_char}, {param_char}, {param_char})",
                (user_id, state, year, month, power_source, forecast_period, predicted_demand, confidence, expected_growth, trend, best_model_used, accuracy, metrics_json)
            )
            inserted_id = cur.lastrowid
        else:
            cur.execute(
                f"INSERT INTO prediction_history (user_id, state, year, month, power_source, forecast_period, "
                f"predicted_demand, confidence, expected_growth, trend, best_model_used, accuracy, metrics) "
                f"VALUES ({param_char}, {param_char}, {param_char}, {param_char}, {param_char}, {param_char}, "
                f"{param_char}, {param_char}, {param_char}, {param_char}, {param_char}, {param_char}, {param_char}) RETURNING id",
                (user_id, state, year, month, power_source, forecast_period, predicted_demand, confidence, expected_growth, trend, best_model_used, accuracy, metrics_json)
            )
            inserted_id = cur.fetchone()[0]
        conn.commit()
        return inserted_id
    except Exception as e:
        conn.rollback()
        print(f"Error saving prediction history: {e}")
        return None
    finally:
        cur.close()
        conn.close()

def get_prediction_history(user_id, limit=50):
    """
    Get prediction history for a specific user.
    """
    conn, is_sqlite = get_connection()
    cur = conn.cursor()
    param_char = '?' if is_sqlite else '%s'
    
    try:
        cur.execute(
            f"SELECT id, state, year, month, power_source, forecast_period, predicted_demand, confidence, "
            f"expected_growth, trend, best_model_used, accuracy, metrics, prediction_time FROM prediction_history "
            f"WHERE user_id = {param_char} ORDER BY prediction_time DESC LIMIT {param_char}",
            (user_id, limit)
        )
        history = cur.fetchall()
        
        history_list = []
        for row in history:
            metrics = row[12]
            if isinstance(metrics, str):
                try:
                    metrics = json.loads(metrics)
                except json.JSONDecodeError:
                    metrics = {}
            
            prediction_time = row[13]
            if isinstance(prediction_time, str):
                try:
                    if '.' in prediction_time:
                        prediction_time = datetime.strptime(prediction_time, '%Y-%m-%d %H:%M:%S.%f')
                    else:
                        prediction_time = datetime.strptime(prediction_time, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    prediction_time = datetime.now()
            
            history_list.append({
                'id': row[0],
                'state': row[1],
                'year': row[2],
                'month': row[3],
                'power_source': row[4],
                'forecast_period': row[5],
                'predicted_demand': row[6],
                'confidence': row[7],
                'expected_growth': row[8],
                'trend': row[9],
                'best_model_used': row[10],
                'accuracy': row[11],
                'metrics': metrics,
                'prediction_time': prediction_time
            })
        return history_list
    except Exception as e:
        print(f"Error retrieving prediction history: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def get_user_by_username(username):
    """
    Retrieve user credentials by username.
    """
    conn, is_sqlite = get_connection()
    cur = conn.cursor()
    param_char = '?' if is_sqlite else '%s'
    
    try:
        cur.execute(f"SELECT id, username, password FROM users WHERE username = {param_char}", (username,))
        row = cur.fetchone()
        if row:
            return {'id': row[0], 'username': row[1], 'password': row[2]}
        return None
    except Exception as e:
        print(f"Error getting user by username: {e}")
        return None
    finally:
        cur.close()
        conn.close()

def create_user(username, hashed_password):
    """
    Create a new user, returning the new user's ID.
    Handles differences in SQLite vs PostgreSQL auto-increment returns.
    """
    conn, is_sqlite = get_connection()
    cur = conn.cursor()
    param_char = '?' if is_sqlite else '%s'
    
    try:
        if is_sqlite:
            cur.execute(
                f"INSERT INTO users (username, password) VALUES (?, ?)",
                (username, hashed_password)
            )
            user_id = cur.lastrowid
        else:
            cur.execute(
                f"INSERT INTO users (username, password) VALUES (%s, %s) RETURNING id",
                (username, hashed_password)
            )
            user_id = cur.fetchone()[0]
        conn.commit()
        return user_id
    except Exception as e:
        conn.rollback()
        print(f"Error creating user: {e}")
        return None
    finally:
        cur.close()
        conn.close()

