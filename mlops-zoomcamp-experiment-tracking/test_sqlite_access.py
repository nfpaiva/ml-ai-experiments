import sqlite3

def test_sqlite_access(db_path):
    try:
        conn = sqlite3.connect(db_path)
        print("Successfully connected to the database.")
        conn.close()
    except sqlite3.OperationalError as e:
        print(f"OperationalError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    db_path = "/nuno_paiva_nos_pt/ml-ai-experiments/mlops-zoomcamp-experiment-tracking/mlflow/mlflow.db"
    test_sqlite_access(db_path)
