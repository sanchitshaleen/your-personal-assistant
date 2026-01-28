"""
Health check script to verify all services are operational before startup.
Run this on application initialization to catch connection issues early.
"""

import os
import sys
import time
from typing import Tuple, List
from src.core.logger import get_logger

log = get_logger(name="core_health")


def check_postgres() -> Tuple[bool, str]:
    """Check PostgreSQL database connection."""
    try:
        import psycopg2
        from dotenv import load_dotenv
        
        load_dotenv()
        
        db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'chat_db'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        }
        
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        conn.close()
        
        return True, "✓ PostgreSQL connected successfully"
    except Exception as e:
        return False, f"✗ PostgreSQL connection failed: {str(e)}"


def check_redis() -> Tuple[bool, str]:
    """Check Redis connection."""
    try:
        import redis
        from dotenv import load_dotenv
        
        load_dotenv()
        
        redis_url = os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/1')
        # Extract host and port from URL
        redis_client = redis.from_url(redis_url)
        redis_client.ping()
        
        return True, "✓ Redis connected successfully"
    except Exception as e:
        return False, f"✗ Redis connection failed: {str(e)}"


def check_qdrant() -> Tuple[bool, str]:
    """Check Qdrant vector database connection."""
    try:
        import requests
        
        qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6335')
        response = requests.get(f"{qdrant_url}/collections", timeout=5)
        
        if response.status_code == 200:
            return True, "✓ Qdrant connected successfully"
        else:
            return False, f"✗ Qdrant returned status {response.status_code}"
    except Exception as e:
        return False, f"✗ Qdrant connection failed: {str(e)}"


def check_ollama() -> Tuple[bool, str]:
    """Check Ollama LLM service connection."""
    try:
        import requests
        
        ollama_url = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        
        if response.status_code == 200:
            return True, "✓ Ollama connected successfully"
        else:
            return False, f"✗ Ollama returned status {response.status_code}"
    except Exception as e:
        return False, f"✗ Ollama connection failed: {str(e)}"


def check_schema() -> Tuple[bool, str]:
    """Check if database schema has required columns."""
    try:
        import psycopg2
        from dotenv import load_dotenv
        
        load_dotenv()
        
        db_config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'chat_db'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        }
        
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        # Check for required columns in uploads table
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'uploads'
        """)
        columns = [row[0] for row in cur.fetchall()]
        
        required_cols = ['embedding_status', 'source']
        missing_cols = [col for col in required_cols if col not in columns]
        
        conn.close()
        
        if missing_cols:
            return False, f"✗ Schema missing columns: {missing_cols}"
        else:
            return True, "✓ Database schema is up to date"
    except Exception as e:
        return False, f"✗ Schema check failed: {str(e)}"


def run_health_checks(verbose: bool = True) -> bool:
    """
    Run all health checks and report results.
    
    Args:
        verbose: If True, print detailed output
        
    Returns:
        bool: True if all checks pass, False otherwise
    """
    checks = [
        ("PostgreSQL", check_postgres),
        ("Redis", check_redis),
        ("Qdrant", check_qdrant),
        ("Ollama", check_ollama),
        ("Database Schema", check_schema),
    ]
    
    results: List[Tuple[str, bool, str]] = []
    
    if verbose:
        log.info("=" * 60)
        log.info("STARTUP HEALTH CHECK")
        log.info("=" * 60)
    
    for service_name, check_func in checks:
        try:
            success, message = check_func()
            results.append((service_name, success, message))
            
            if verbose:
                log.info(f"\n{service_name}:")
                log.info(f"  {message}")
                
        except Exception as e:
            results.append((service_name, False, str(e)))
            if verbose:
                log.error(f"\n{service_name}:")
                log.error(f"  ✗ Unexpected error: {str(e)}")
    
    # Summary
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    if verbose:
        log.info("\n" + "=" * 60)
        log.info(f"RESULT: {passed}/{total} checks passed")
        log.info("=" * 60 + "\n")
    
    all_passed = passed == total
    
    if not all_passed and verbose:
        log.warning("⚠️  Some services are not available!")
        log.warning("Failed services:")
        for service, success, message in results:
            if not success:
                log.warning(f"  - {service}: {message}")
        log.warning("\nTroubleshooting:")
        log.warning("  1. Run: docker-compose -f docker-compose.dev.yml ps")
        log.warning("  2. Check logs: docker-compose -f docker-compose.dev.yml logs [service]")
        log.warning("  3. Restart: docker-compose -f docker-compose.dev.yml down && up -d")
    
    return all_passed


if __name__ == "__main__":
    # Allow running as standalone script
    success = run_health_checks(verbose=True)
    sys.exit(0 if success else 1)
