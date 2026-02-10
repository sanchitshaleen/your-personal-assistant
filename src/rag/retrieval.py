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
        Initialize BM25 + Semantic hybrid retriever with RRF.
        
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
            
            if not all_docs:
                log.warning("No documents found for BM25 index.")
                self.bm25_retriever = None
                return

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
        """
        log.info(f"Starting RRF hybrid retrieval for: '{query[:80]}'")
        if selected_files:
            log.info(f"Filtering to selected files: {selected_files}")
        
        # Step 1: BM25 retrieval
        bm25_docs = self._retrieve_bm25(query)
        
        # Step 2: Semantic retrieval (Qdrant)
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
            log.info(f"  {i}. [RRF={rrf_score:.4f}, BM25={bm25_rank}, Sem={semantic_rank}, File={filename}] {content}...")
        
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
        Combine BM25 and Semantic rankings using RRF (Reciprocal Rank Fusion).
        
        Args:
            bm25_docs: Documents from BM25 retriever
            semantic_docs: Documents from Semantic retriever
            k: Constant for RRF (typically 60)
        
        Returns:
            Dict mapping document content to {'doc': doc, 'rrf_score': float, ...}
        """
        combined = {}
        
        # Helper to process a list of docs
        def process_docs(docs, content_key_len=200, source_name="bm25"):
            for rank, doc in enumerate(docs, start=1):
                doc_key = doc.page_content[:content_key_len]
                if doc_key not in combined:
                    combined[doc_key] = {
                        'doc': doc, 
                        'bm25_rank': None, 
                        'semantic_rank': None,
                        'rrf_score': 0.0
                    }
                combined[doc_key][f'{source_name}_rank'] = rank

        # Process all sources
        process_docs(bm25_docs, source_name="bm25")
        process_docs(semantic_docs, source_name="semantic")
        
        # Calculate RRF scores
        for key in combined:
            rrf_score = 0.0
            info = combined[key]
            
            if info['bm25_rank']: rrf_score += 1.0 / (k + info['bm25_rank'])
            if info['semantic_rank']: rrf_score += 1.0 / (k + info['semantic_rank'])
            
            combined[key]['rrf_score'] = rrf_score
        
        return combined


class Neo4jVectorRetriever:
    """
    Retriever that uses Neo4j Vector Index to find relevant nodes.
    Separated from the Hybrid Retriever to allow for Agentic selection.
    """
    def __init__(self, neo4j_graph, embeddings, k: int = 5):
        self.neo4j_graph = neo4j_graph
        self.embeddings = embeddings
        self.k = k
    
    def get_relevant_documents(self, query: str, selected_files: List[str] = None) -> List[Document]:
        """Retrieve documents using Neo4j Graph Vector Index, optionally filtering by filename."""
        if not self.neo4j_graph:
            log.warning("Neo4j Graph not initialized.")
            return []
        
        try:
            # 1. Calculate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # 2. Key: Neo4j Vector Index search isn't easily compatible with pre-filtering in LangChain's existing wrapper
            # But since we use raw Cypher here, we can simple post-filter or try to pre-filter if index allows.
            # Neo4j vector index query `db.index.vector.queryNodes` doesn't support filter directly in the procedure call.
            # We must fetch more and filter, or use the new `queryNodes` signature in 5.x if available, 
            # Or use post-filtering (WHERE node.source in list) which is less efficient if k is small.
            
            # Strategy: Fetch k*4 candidates, then filter.
            
            k_to_fetch = self.k * 2
            if selected_files:
                k_to_fetch = self.k * 5 # Fetch more if filtering
            
            query_cypher = """
            CALL db.index.vector.queryNodes('movie_text_embeddings', $k, $embedding)
            YIELD node, score
            """
            
            params = {
                "k": k_to_fetch, 
                "embedding": query_embedding
            }
            
            # Add filtering
            if selected_files:
                # Assuming 'source' property holds filename or path
                # We need to match how it's stored. GraphIndexer stores 'source' in metadata.
                # In add_graph_documents, include_source=True adds 'source' property to Document node.
                query_cypher += """
                WHERE any(file IN $selected_files WHERE node.source CONTAINS file)
                """
                params["selected_files"] = selected_files
            
            query_cypher += """
            // Traversal: Find connected entities
            MATCH (node)-[r:MENTIONS]->(e)
            WITH node, score, collect(e) as entity_nodes, collect(r) as rels
            
            // Format Output
            RETURN 
                "Related Entities: " + apoc.text.join([e in entity_nodes | e.id], ", ") + "\n\n" + node.text AS text,
                node.source AS source, 
                score, 
                elementId(node) as id,
                {
                    nodes: [n in entity_nodes + [node] | {id: elementId(n), labels: labels(n), properties: properties(n)}],
                    relationships: [rel in rels | {id: elementId(rel), type: type(rel), startNode: elementId(startNode(rel)), endNode: elementId(endNode(rel)), properties: properties(rel)}]
                } as graph_data
            LIMIT $final_k
            """
            params["final_k"] = self.k
            
            results = self.neo4j_graph.query(query_cypher, params=params)
            
            # 3. Convert to Documents
            docs = []
            for res in results:
                page_content = res.get('text', '')
                source = res.get('source', 'graph')
                metadata = {
                    'source': source, 
                    'filename': source if source != 'graph' else 'neo4j_graph', 
                    'graph_score': res.get('score'), 
                    'id': res.get('id'),
                    'graph_data': res.get('graph_data') # Include graph structure
                }
                doc = Document(page_content=page_content, metadata=metadata)
                docs.append(doc)
                
            log.info(f"Graph vector retrieval returned {len(docs)} documents (filtered={bool(selected_files)})")
            return docs

        except Exception as e:
            log.error(f"Graph retrieval failed: {e}")
            return []
            
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)
