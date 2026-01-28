"""
semantic_cache.py - Semantic caching layer for RAG responses.

This module implements semantic caching to reduce latency and computational cost
by caching query results and reusing them for semantically similar queries.

Uses Redis for persistent storage and cosine similarity for semantic matching.
Includes automatic cache invalidation when backend code changes.
"""

import redis
import json
import numpy as np
import time
from typing import Optional, Dict, Any
from src.core.logger import get_logger

log = get_logger(name="semantic_cache")


class SemanticCache:
    """Semantic caching layer using Redis with similarity thresholds.
    
    This class implements semantic caching by:
    1. Converting queries to embeddings using the same model as document embeddings
    2. Computing cosine similarity between new and cached queries
    3. Returning cached results for semantically similar queries
    4. Storing new query results in Redis with automatic expiration
    
    Attributes:
        embeddings: The embedding model (mxbai-embed-large)
        redis_client: Redis connection for persistent cache storage
        similarity_threshold: Minimum cosine similarity (0-1) to use cached result
        cache_prefix: Redis key namespace for semantic cache entries (includes version hash)
        version_hash: Hash of critical backend files for invalidation detection
    """
    
    def __init__(
        self,
        embeddings_model,
        redis_client: redis.Redis,
        similarity_threshold: float = 0.85,
        cache_prefix: str = "semantic_cache:",
        version_hash: Optional[str] = None
    ):
        """Initialize the semantic cache.
        
        Args:
            embeddings_model: The embedding model instance (e.g., OllamaEmbeddings)
            redis_client: Redis client for cache storage
            similarity_threshold: Minimum similarity score (0-1) to use cached result
            cache_prefix: Base prefix for Redis keys to avoid collisions
            version_hash: Code version hash for cache invalidation (optional)
        """
        self.embeddings = embeddings_model
        self.redis_client = redis_client
        self.similarity_threshold = similarity_threshold
        self.version_hash = version_hash
        
        # Build versioned prefix if version_hash provided
        if version_hash:
            self.cache_prefix = f"{cache_prefix}v_{version_hash}:"
        else:
            self.cache_prefix = cache_prefix
        
        log.info(
            f"Initialized SemanticCache with threshold={similarity_threshold}, "
            f"prefix={self.cache_prefix}"
        )
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding vector for a query.
        
        Args:
            query: The query text to embed
            
        Returns:
            np.ndarray: The embedding vector (typically 1024 dimensions)
        """
        try:
            embedding = self.embeddings.embed_query(query)
            return np.array(embedding)
        except Exception as e:
            log.error(f"Error generating embedding for query: {e}")
            return None
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Cosine similarity measures the angle between vectors in high-dimensional space.
        Values range from -1 to 1, where:
        - 1.0 = identical direction (perfectly similar)
        - 0.5 = moderate similarity
        - 0.0 = orthogonal (completely different)
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            float: Cosine similarity score between -1 and 1
        """
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            log.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_cached_result(
        self,
        query: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Check if a semantically similar query result is cached.
        
        Algorithm:
        1. Generate embedding for the input query
        2. Retrieve all cached queries for this user from Redis
        3. Compare input embedding with each cached query embedding
        4. Return the best match if similarity exceeds threshold
        5. Return None if no match found
        
        Args:
            query: The input query string
            user_id: The user ID to scope cache to (ensures privacy)
            
        Returns:
            Dict with keys:
                - 'result': The cached RAG response
                - 'similarity': The similarity score (0-1)
                - 'cached_query': The original cached query text
            or None if no suitable match found
        """
        try:
            if self.redis_client is None:
                log.warning("Redis client not available, cannot check cache")
                return None
            
            # Generate embedding for input query
            query_embedding = self.get_query_embedding(query)
            if query_embedding is None:
                log.warning(f"Could not generate embedding for query, skipping cache lookup")
                return None
            
            # Get all cached queries for this user
            cache_key = f"{self.cache_prefix}{user_id}:queries"
            cached_queries = self.redis_client.hgetall(cache_key)
            
            if not cached_queries:
                log.debug(f"No cached queries found for user {user_id}")
                return None
            
            best_match = None
            best_similarity = 0.0
            
            # Compare with each cached query
            for cached_query_str, cached_data_json in cached_queries.items():
                try:
                    cached_data = json.loads(cached_data_json)
                    cached_embedding = np.array(cached_data.get('embedding', []))
                    
                    if len(cached_embedding) == 0:
                        continue
                    
                    # Calculate similarity
                    sim_score = self.similarity(query_embedding, cached_embedding)
                    
                    # Keep track of best match
                    if sim_score > best_similarity:
                        best_similarity = sim_score
                        best_match = (cached_query_str, cached_data, sim_score)
                
                except json.JSONDecodeError as e:
                    log.warning(f"Error parsing cached query: {e}")
                    continue
            
            # Return best match if it meets threshold
            if best_match and best_similarity >= self.similarity_threshold:
                cached_query_str, cached_data, sim_score = best_match
                log.info(
                    f"Cache HIT for user {user_id}. "
                    f"Similarity: {sim_score:.3f}, "
                    f"Original query: '{cached_query_str}'"
                )
                return {
                    'result': cached_data.get('result'),
                    'similarity': sim_score,
                    'cached_query': cached_query_str
                }
            
            if best_similarity > 0:
                log.debug(
                    f"Cache MISS for user {user_id}. "
                    f"Best similarity: {best_similarity:.3f} (threshold: {self.similarity_threshold})"
                )
            
            return None
        
        except Exception as e:
            log.error(f"Error checking semantic cache: {e}")
            return None
    
    def cache_result(
        self,
        query: str,
        result: Dict[str, Any],
        user_id: str,
        ttl: int = 86400
    ) -> bool:
        """Cache a query result with its embedding for future reuse.
        
        Stores the query, its embedding, and the RAG result in Redis with:
        - User-scoped key for privacy
        - TTL-based expiration to prevent unbounded growth
        - JSON serialization for language-agnostic storage
        
        Args:
            query: The query text
            result: The RAG response (dict with answer, sources, metadata)
            user_id: The user ID for scoping (ensures privacy)
            ttl: Time-to-live in seconds (default: 24 hours)
            
        Returns:
            bool: True if successfully cached, False otherwise
        """
        try:
            if self.redis_client is None:
                log.warning("Redis client not available, cannot cache result")
                return False
            
            # Generate embedding for this query
            query_embedding = self.get_query_embedding(query)
            if query_embedding is None:
                log.warning("Could not generate embedding, skipping cache storage")
                return False
            
            # Package data for storage
            cache_data = {
                'query': query,
                'embedding': query_embedding.tolist(),  # Convert numpy array to list for JSON
                'result': result,
                'timestamp': int(time.time())
            }
            
            # Store in Redis as hash field
            cache_key = f"{self.cache_prefix}{user_id}:queries"
            self.redis_client.hset(
                cache_key,
                query,
                json.dumps(cache_data)
            )
            
            # Set expiration time
            self.redis_client.expire(cache_key, ttl)
            
            log.info(
                f"Cached result for user {user_id}. "
                f"TTL: {ttl}s, Query: '{query[:50]}...'"
            )
            return True
        
        except Exception as e:
            log.error(f"Error caching result: {e}")
            return False
    
    def clear_user_cache(self, user_id: str) -> bool:
        """Clear all cached queries for a specific user.
        
        Args:
            user_id: The user ID whose cache to clear
            
        Returns:
            bool: True if successfully cleared, False otherwise
        """
        try:
            if self.redis_client is None:
                log.warning("Redis client not available, cannot clear cache")
                return False
            
            cache_key = f"{self.cache_prefix}{user_id}:queries"
            deleted = self.redis_client.delete(cache_key)
            
            log.info(f"Cleared semantic cache for user {user_id}")
            return deleted > 0
        
        except Exception as e:
            log.error(f"Error clearing user cache: {e}")
            return False
    
    def get_cache_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about cached queries for a user.
        
        Args:
            user_id: The user ID to get stats for
            
        Returns:
            Dict with keys:
                - 'num_cached_queries': Number of queries cached
                - 'cache_size_bytes': Approximate size in Redis
                - 'oldest_timestamp': Timestamp of oldest cache entry
                - 'newest_timestamp': Timestamp of newest cache entry
        """
        try:
            if self.redis_client is None:
                return {'error': 'Redis not available'}
            
            cache_key = f"{self.cache_prefix}{user_id}:queries"
            cached_queries = self.redis_client.hgetall(cache_key)
            
            if not cached_queries:
                return {
                    'num_cached_queries': 0,
                    'cache_size_bytes': 0,
                    'oldest_timestamp': None,
                    'newest_timestamp': None
                }
            
            timestamps = []
            total_size = 0
            
            for cached_data_json in cached_queries.values():
                try:
                    cached_data = json.loads(cached_data_json)
                    total_size += len(cached_data_json)
                    timestamps.append(cached_data.get('timestamp', 0))
                except:
                    pass
            
            return {
                'num_cached_queries': len(cached_queries),
                'cache_size_bytes': total_size,
                'oldest_timestamp': min(timestamps) if timestamps else None,
                'newest_timestamp': max(timestamps) if timestamps else None
            }
        
        except Exception as e:
            log.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
