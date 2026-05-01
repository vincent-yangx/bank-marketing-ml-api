import json
import pymysql

DB_config = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "password",
    "database": "bank_ml",
    "charset": "utf8mb4",
}

def get_db_connection():
    try:
        connection = pymysql.connect(**DB_config)
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise
    
def init_db():
    create_table_sql = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    input_features JSON,
    prediction VARCHAR(10),
    prob_no FLOAT,
    prob_yes FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""
    connection = get_db_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(create_table_sql)
            connection.commit()
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise
        finally:
            connection.close()

def insert_prediction_log(input_features, prediction, probability):
    insert_sql = """
    INSERT INTO prediction_logs(
        input_features,
        prediction,
        prob_no,
        prob_yes
    )
    VALUES (%s, %s, %s, %s);
    """

    prob_no = probability.get("no")
    prob_yes = probability.get("yes")

    connection = get_db_connection()
    if connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    insert_sql,
                    (
                        json.dumps(input_features),
                        prediction,
                        prob_no,
                        prob_yes
                    )
                )
            connection.commit()
        except Exception as e:
            print(f"Error inserting prediction log: {e}")
        finally:
            connection.close()

def get_recent_prediction_logs(limit = 10):
    select_sql = """
    SELECT id, input_features, prediction, prob_no, prob_yes, 
    created_at
    FROM prediction_logs
    ORDER BY created_at DESC
    LIMIT %s;
    """
    connection = get_db_connection()
    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(select_sql, (limit,))
            logs = cursor.fetchall()
        return logs
    except Exception as e:
        print(f"Error fetching prediction logs; {e}")
        raise
    finally:
        connection.close()
    
    