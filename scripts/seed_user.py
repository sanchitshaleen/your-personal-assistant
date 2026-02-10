import sys
import os
import bcrypt
import psycopg2
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import settings as config

def seed_user():
    try:
        conn = psycopg2.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            database=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD
        )
        cur = conn.cursor()
        
        cst_time = "2024-01-01 00:00:00"
        hashed = bcrypt.hashpw("password".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        cur.execute("""
            INSERT INTO users (user_id, name, password_hash, last_login)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id) DO NOTHING
        """, ("default_user", "Default User", hashed, cst_time))
        
        conn.commit()
        conn.close()
        print("User seeded successfully.")
    except Exception as e:
        print(f"Error seeding user: {e}")

if __name__ == "__main__":
    seed_user()
