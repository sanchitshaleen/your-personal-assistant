"""BM25 + Semantic Hybrid Retrieval with RRF (Reciprocal Rank Fusion)

This module implements a hybrid retrieval approach combining:
1. BM25 (Sparse Lexical Matching) - Built-in LangChain
2. Semantic (Dense Vector Search) - Qdrant
3. RRF (Reciprocal Rank Fusion) - Combines rankings from both sources

RRF treats each retriever's ranking as equal and fuses them by position,
avoiding the scale sensitivity of weighted averaging approaches.
"""

import os
import sys
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import numpy as np

from src.core.logger import get_logger
log = get_logger(name="bm25_semantic_retriever")


class BM25SemanticHybridRetriever:
    """Combines BM25 lexical search with semantic vector search for deterministic hybrid retrieval."""
    
    def __init__(
        self,
        qdrant_client,
        collection_name: str,
        embeddings: Embeddings,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        k: int = 5
    ):
        """
        Initialize BM25 + Semantic hybrid retriever with RRF (Reciprocal Rank Fusion).
        
        Args:
            qdrant_client: Qdrant client instance
            collection_name: Qdrant collection name
            embeddings: Embeddings model for dense vectors
            bm25_weight: Deprecated (kept for backward compatibility, not used in RRF)
            semantic_weight: Deprecated (kept for backward compatibility, not used in RRF)
            k: Number of final results to return
        """
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.k = k
        
        # Note: bm25_weight and semantic_weight are no longer used with RRF scoring
        # RRF treats both ranking sources equally and combines them by rank position
        
        # Initialize BM25 index
        self._initialize_bm25_index()
        
        log.info(f"BM25SemanticHybridRetriever initialized with RRF scoring: k={k}")
    
    def _initialize_bm25_index(self):
        """Initialize BM25 index from all documents in collection."""
        try:
            from langchain.retrievers import BM25Retriever
            
            # Fetch all documents from Qdrant
            all_docs = self._fetch_all_documents()
            log.info(f"Fetched {len(all_docs)} documents for BM25 indexing")
            
            # Create BM25 retriever
            self.bm25_retriever = BM25Retriever.from_documents(
                documents=all_docs,
                k=self.k * 2  # Get more for fusion
            )
            log.info("BM25 index initialized successfully")
        except ImportError:
            log.warning("BM25Retriever not available. Install 'rank_bm25' package.")
            self.bm25_retriever = None
        except Exception as e:
            log.error(f"Failed to initialize BM25: {e}")
            self.bm25_retriever = None
    
    def _fetch_all_documents(self) -> List[Document]:
        """Fetch all documents from Qdrant collection."""
        try:
            docs = []
            offset = 0
            limit = 100
            
            while True:
                # Scroll through collection
                points, next_offset = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset
                )
                
                if not points:
                    break
                
                for point in points:
                    if hasattr(point, 'payload'):
                        payload = point.payload
                        page_content = payload.get('page_content', 
                                                  payload.get('content', str(payload)))
                        metadata = {k: v for k, v in payload.items() 
                                   if k not in ['page_content', 'content']}
                        doc = Document(page_content=page_content, metadata=metadata)
                        docs.append(doc)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            log.info(f"Fetched {len(docs)} documents from Qdrant")
            return docs
        
        except Exception as e:
            log.error(f"Error fetching documents: {e}")
            return []
    
    def retrieve_hybrid(self, query: str, selected_files: List[str] = None) -> List[Document]:
        """
        Perform hybrid retrieval using RRF (Reciprocal Rank Fusion).
        
        RRF combines rankings from BM25 and Semantic retrievers using:
        RRF(d) = sum(1 / (k + rank)) for each ranker
        
        Args:
            query: User query string
            selected_files: Optional list of filenames to filter results. If provided, only docs from these files are returned.
        
        Returns:
            List of top-k documents sorted by RRF score
        """
        log.info(f"Starting RRF hybrid retrieval for: '{query[:80]}'")
        if selected_files:
            log.info(f"Filtering to selected files: {selected_files}")
        
        # Step 1: BM25 retrieval
        bm25_docs = self._retrieve_bm25(query)
        
        # Step 2: Semantic retrieval
        semantic_docs = self._retrieve_semantic(query)
        
        # Step 3: Combine using RRF
        combined = self._combine_scores_rrf(bm25_docs, semantic_docs)
        
        # Step 4: Filter by selected files if provided
        if selected_files:
            combined = {k: v for k, v in combined.items() if v['doc'].metadata.get('filename') in selected_files}
            log.info(f"After file filtering: {len(combined)} documents remaining")
        
        # Step 5: Sort and return top-k
        final_docs = sorted(combined.items(), key=lambda x: x[1]['rrf_score'], reverse=True)[:self.k]
        
        # Log ranking details for debugging
        log.info(f"RRF Top Results:")
        for i, (key, info) in enumerate(final_docs[:3], 1):
            doc = info['doc']
            bm25_rank = info.get('bm25_rank')
            semantic_rank = info.get('semantic_rank')
            rrf_score = info['rrf_score']
            filename = doc.metadata.get('filename', 'unknown')
            content = doc.page_content[:80]
            log.info(f"  {i}. [RRF={rrf_score:.4f}, BM25_rank={bm25_rank}, Semantic_rank={semantic_rank}, File={filename}] {content}...")
        
        # Extract actual Document objects from the sorted results
        final_docs = [info['doc'] for key, info in final_docs]
        
        # NOTE: RRF avoids issues with weighted averaging by using ranking positions
        # Documents that rank well in EITHER BM25 OR Semantic get boosted
        # This is more robust than weighted averaging for combining diverse ranking signals
        
        log.info(f"RRF hybrid retrieval returned {len(final_docs)} documents")
        return final_docs
    
    def _retrieve_bm25(self, query: str) -> List[Document]:
        """Retrieve documents using BM25."""
        if self.bm25_retriever is None:
            return []
        
        try:
            docs = self.bm25_retriever.invoke(query)
            log.info(f"BM25 retrieval returned {len(docs)} documents")
            return docs
        except Exception as e:
            log.error(f"BM25 retrieval failed: {e}")
            return []
    
    def _retrieve_semantic(self, query: str) -> List[Document]:
        """Retrieve documents using semantic (dense vector) search."""
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in Qdrant
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=self.k * 2
            )
            
            # Convert to documents
            docs = []
            for result in results:
                if hasattr(result, 'payload'):
                    payload = result.payload
                    page_content = payload.get('page_content', 
                                              payload.get('content', str(payload)))
                    metadata = {k: v for k, v in payload.items() 
                               if k not in ['page_content', 'content']}
                    metadata['_id'] = getattr(result, 'id', None)
                    metadata['semantic_score'] = float(result.score) if hasattr(result, 'score') else 0.0
                    doc = Document(page_content=page_content, metadata=metadata)
                    docs.append(doc)
            
            log.info(f"Semantic retrieval returned {len(docs)} documents")
            return docs
        
        except Exception as e:
            log.error(f"Semantic retrieval failed: {e}")
            return []
    
    def _combine_scores_rrf(
        self,
        bm25_docs: List[Document],
        semantic_docs: List[Document],
        k: int = 60
    ) -> Dict[str, Dict[str, float]]:
        """
        Combine BM25 and semantic rankings using RRF (Reciprocal Rank Fusion).
        
        RRF formula: score(d) = sum(1 / (k + rank(d, retriever)))
        
        This approach:
        - Treats each retriever as a ranking source (rank 1, 2, 3, ...)
        - Gives equal weight to good rankings from either source
        - Is more robust than score averaging to scale differences
        
        Args:
            bm25_docs: Documents from BM25 retriever in rank order
            semantic_docs: Documents from semantic retriever in rank order
            k: Constant for RRF (typically 60), prevents rank=0 issues
        
        Returns:
            Dict mapping document content to {'doc': doc, 'bm25_rank': int, 'semantic_rank': int, 'rrf_score': float}
        """
        combined = {}
        
        # Process BM25 results with ranks (1-indexed)
        for rank, doc in enumerate(bm25_docs, start=1):
            doc_key = doc.page_content[:200]  # Use content as key
            
            if doc_key not in combined:
                combined[doc_key] = {
                    'doc': doc, 
                    'bm25_rank': None, 
                    'semantic_rank': None,
                    'rrf_score': 0.0
                }
            
            combined[doc_key]['bm25_rank'] = rank
        
        # Process semantic results with ranks (1-indexed)
        for rank, doc in enumerate(semantic_docs, start=1):
            doc_key = doc.page_content[:200]  # Use content as key
            
            if doc_key not in combined:
                combined[doc_key] = {
                    'doc': doc, 
                    'bm25_rank': None, 
                    'semantic_rank': None,
                    'rrf_score': 0.0
                }
            
            combined[doc_key]['semantic_rank'] = rank
        
        # Calculate RRF scores
        for key in combined:
            rrf_score = 0.0
            
            # Add contribution from BM25 rank if available
            if combined[key]['bm25_rank'] is not None:
                rrf_score += 1.0 / (k + combined[key]['bm25_rank'])
            
            # Add contribution from semantic rank if available
            if combined[key]['semantic_rank'] is not None:
                rrf_score += 1.0 / (k + combined[key]['semantic_rank'])
            
            combined[key]['rrf_score'] = rrf_score
        
        return combined
