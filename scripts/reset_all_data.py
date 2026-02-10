import os
import sys
import psycopg2
import bcrypt
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.logger import get_logger
from config import settings as config

log = get_logger("data_reset")

# DB Config from env or defaults (matching src/core/database.py)
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'postgres'), # Default inside container
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'chat_db'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
}

def reset_postgres():
    log.info("--- 1. Cleaning PostgreSQL ---")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Order matters due to foreign keys (embeddings -> uploads -> users)
        tables = ["embeddings", "uploads", "users"] 
        for table in tables:
            cur.execute(f"TRUNCATE TABLE {table} CASCADE;")
            log.info(f"   ✓ Truncated table: {table}")
            
        conn.commit()
        
        # Re-seed default user
        log.info("   + Re-seeding default user...")
        cst_time = "2024-01-01 00:00:00" 
        hashed = bcrypt.hashpw("password".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cur.execute("""
            INSERT INTO users (user_id, name, password_hash, last_login)
            VALUES (%s, %s, %s, %s)
        """, ("default_user", "Default User", hashed, cst_time))
        conn.commit()
        log.info("   ✓ Re-created 'default_user'")
        
        conn.close()
        log.info("✓ PostgreSQL cleaned and seeded.")
    except Exception as e:
        log.error(f"✗ Failed to clean Postgres: {e}")
        # Investigate connection if failure
        log.error(f"   (Config used: {DB_CONFIG})")

def reset_qdrant():
    log.info("--- 2. Cleaning Qdrant ---")
    try:
        from qdrant_client import QdrantClient
        # Use env vars or config
        host = os.getenv("QDRANT_HOST", "qdrant")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        
        client = QdrantClient(host=host, port=port)
        
        collections = config.QDRANT_COLLECTION_NAME 
        cols = [collections, "document_summaries"]
        
        for col_name in cols:
            try:
                client.delete_collection(col_name)
                log.info(f"   ✓ Deleted collection: {col_name}")
            except Exception as e:
                # API might raise error if collection doesn't exist
                log.warning(f"   - Collection {col_name} check: {e}")
                
        log.info("✓ Qdrant cleaned.")
    except Exception as e:
        log.error(f"✗ Failed to clean Qdrant: {e}")

def reset_neo4j():
    log.info("--- 3. Cleaning Neo4j ---")
    if not config.GRAPH_INGESTION_ENABLED:
        log.info("   - Graph ingestion disabled, skipping Neo4j.")
        return

    try:
        from langchain_community.graphs import Neo4jGraph
        # Neo4jGraph is deprecated but we use it as imports are available
        graph = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD
        )
        
        graph.query("MATCH (n) DETACH DELETE n")
        
        # Verify
        count = graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        if count == 0:
            log.info(f"   ✓ Deleted all nodes. Count is now 0.")
        else:
            log.warning(f"   ! Database not empty. Count: {count}")
            
        log.info("✓ Neo4j cleaned.")
    except Exception as e:
        log.error(f"✗ Failed to clean Neo4j: {e}")

if __name__ == "__main__":
    print("WARNING: This will wipe ALL data from Postgres, Qdrant, and Neo4j.")
    
    reset_postgres()
    reset_qdrant()
    reset_neo4j()
    
    print("\n✓ CLEAN SLATE COMPLETE.")
