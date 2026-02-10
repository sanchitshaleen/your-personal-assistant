import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.logger import get_logger
from config import settings as config

log = get_logger("vector_store_reset")

def clear_qdrant():
    log.info("--- Cleaning Qdrant ---")
    try:
        from qdrant_client import QdrantClient
        host = os.getenv("QDRANT_HOST", "qdrant")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        
        client = QdrantClient(host=host, port=port)
        
        collections = [config.QDRANT_COLLECTION_NAME, "document_summaries"]
        
        for col_name in collections:
            try:
                client.delete_collection(col_name)
                log.info(f"   ✓ Deleted collection: {col_name}")
            except Exception as e:
                log.warning(f"   - Collection {col_name}: {e}")
                
        log.info("✓ Qdrant cleaned.")
    except Exception as e:
        log.error(f"✗ Failed to clean Qdrant: {e}")

def clear_neo4j():
    log.info("--- Cleaning Neo4j ---")
    if not config.GRAPH_INGESTION_ENABLED:
        log.info("   - Graph ingestion disabled, skipping Neo4j.")
        return

    try:
        from langchain_community.graphs import Neo4jGraph
        graph = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD
        )
        
        graph.query("MATCH (n) DETACH DELETE n")
        
        count = graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        if count == 0:
            log.info(f"   ✓ Deleted all nodes. Count is now 0.")
        else:
            log.warning(f"   ! Database not empty. Count: {count}")
            
        log.info("✓ Neo4j cleaned.")
    except Exception as e:
        log.error(f"✗ Failed to clean Neo4j: {e}")

if __name__ == "__main__":
    print("Clearing Qdrant and Neo4j...")
    clear_qdrant()
    clear_neo4j()
    print("\n✓ Vector stores cleared.")
