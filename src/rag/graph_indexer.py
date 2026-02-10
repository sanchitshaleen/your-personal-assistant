from typing import List, Optional
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from config import settings as config
from src.core import logger

log = logger.get_logger("graph_indexer")

class GraphIndexer:
    """
    Handles indexing of documents into Neo4j Knowledge Graph.
    Uses LLMGraphTransformer to extract entities and relationships.
    """
    
    
    def __init__(self, llm, embeddings=None):
        """
        Initialize GraphIndexer with an LLM and Neo4j connection.
        
        Args:
            llm: The Language Model to use for extraction (e.g., ChatOllama)
            embeddings: Optional Embedding model for vectorizing chunks (e.g., Nomic)
        """
        self.llm = llm
        self.embeddings = embeddings
        self.graph = None
        self.transformer = None
        
        if config.GRAPH_INGESTION_ENABLED:
            try:
                self.graph = Neo4jGraph(
                    url=config.NEO4J_URI,
                    username=config.NEO4J_USERNAME,
                    password=config.NEO4J_PASSWORD
                )
                log.info("✓ Connected to Neo4j Graph Database")
                
                # Check/Create Vector Index if embeddings provided
                if self.embeddings:
                    try:
                        self.graph.query(
                            """
                            CREATE VECTOR INDEX movie_text_embeddings IF NOT EXISTS
                            FOR (n:Document)
                            ON (n.embedding)
                            OPTIONS {indexConfig: {
                              `vector.dimensions`: 768,
                              `vector.similarity_function`: 'cosine'
                            }}
                            """
                        )
                        # Check dimensions - Nomic is 768. 
                        # If using different model, might need to adjust or make dynamic.
                        # For now assume Nomic (768).
                        log.info("✓ Verified Neo4j Vector Index")
                    except Exception as ve:
                        log.warning(f"Could not create vector index: {ve}")

                # Initialize the transformer
                self.transformer = LLMGraphTransformer(
                    llm=self.llm,
                    allowed_nodes=["Person", "Organization", "Location", "Event", "Concept", "Document"],
                    allowed_relationships=["OWNS", "CREATED", "LOCATED_IN", "PART_OF", "RELATED_TO", "MENTIONS"],
                    node_properties=["name", "description"]
                )
                
            except Exception as e:
                log.error(f"✗ Failed to connect to Neo4j: {e}")
                self.graph = None
    
    def index_documents(self, documents: List[Document], batch_size: int = 5, llm_summary=None, qdrant_client=None, user_id="default_user", file_path="unknown") -> bool:
        """
        Extract graph data from documents and store in Neo4j (with embeddings).
        Also creates document summary in Qdrant for planner retrieval.
        
        Args:
            documents: List of LangChain Documents to process
            batch_size: Number of documents to process in one LLM call/transaction
            llm_summary: Optional LLM model for summary generation
            qdrant_client: Optional Qdrant client for storing summaries
            user_id: User ID for the documents
            file_path: Path to the original file being indexed
        """
        if not self.graph or not self.transformer:
            log.warning("Graph ingestion skipped: Neo4j not initialized or disabled")
            return False
            
        log.info(f"Starting graph extraction for {len(documents)} document chunks...")
        
        try:
            total_docs = len(documents)
            
            # Process in batches
            for i in range(0, total_docs, batch_size):
                batch = documents[i : i + batch_size]
                log.info(f"Processing graph batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} ({len(batch)} docs)")
                
                try:
                    # 1. Transform documents to graph documents
                    graph_documents = self.transformer.convert_to_graph_documents(batch)
                    
                    # LOGGING: details about what was extracted
                    if graph_documents:
                        nodes_count = sum(len(d.nodes) for d in graph_documents)
                        rels_count = sum(len(d.relationships) for d in graph_documents)
                        log.info(f"Extracted {nodes_count} nodes and {rels_count} relationships from batch")
                        if nodes_count == 0:
                             log.warning("No nodes extracted! Model might be failing to produce valid JSON.")
                    else:
                        log.warning("No graph documents produced.")

                    # 2. Store in Neo4j
                    self.graph.add_graph_documents(
                        graph_documents,
                        baseEntityLabel=True,
                        include_source=True
                    )
                    
                    # 3. Add Embeddings to Document nodes
                    if self.embeddings:
                        # For each doc in batch, compute embedding and update in Neo4j
                        # Note: add_graph_documents (via creating Document nodes) uses the MD5 of py_content as ID ?
                        # Or it links GraphDocument to Source Document.
                        # We need to match the Text content to find the node.
                        
                        # Optimization: Batch embed
                        texts = [d.page_content for d in batch]
                        embeddings_list = self.embeddings.embed_documents(texts)
                        
                        # Update query
                        # We match by `text` property on `Document` node which langchain-community wrapper sets
                        for j, doc in enumerate(batch):
                            embedding = embeddings_list[j]
                            # Escape text purely for safety, though parameterized query is better
                            # Using parameterized query via self.graph.query
                            query = """
                            MATCH (d:Document {text: $text})
                            SET d.embedding = $embedding
                            """
                            self.graph.query(query, params={"text": doc.page_content, "embedding": embedding})
                            
                    log.info(f"✓ Batch {i//batch_size + 1} stored in Neo4j (with embeddings)")
                    
                except Exception as batch_error:
                    log.error(f"Error processing batch {i}: {batch_error}")

            # 4. CREATE DOCUMENT SUMMARY FOR PLANNER (Added fix)
            if llm_summary and qdrant_client and self.embeddings:
                try:
                    import time
                    log.info(f"Generating summary for Neo4j-indexed file: {file_path}...")
                    from langchain_core.prompts import ChatPromptTemplate
                    from langchain_core.output_parsers import StrOutputParser
                    from qdrant_client.models import PointStruct
                    
                    # Combine first 20000 chars as context
                    full_text = " ".join([d.page_content for d in documents])[:20000]
                    
                    prompt = ChatPromptTemplate.from_template(
                        "Summarize the following document content in 3-5 sentences. "
                        "Focus on the main topics, entities, and purpose of the document so a retrieval system knows when to select it. "
                        "Mention that this is stored in the Knowledge Graph database (Neo4j).\n\n"
                        "Content:\n{content}"
                    )
                    chain = prompt | llm_summary | StrOutputParser()
                    summary_text = chain.invoke({"content": full_text})
                    log.info(f"Generated summary: {summary_text}")
                    
                    # Embed and store summary
                    summary_embedding = self.embeddings.embed_query(summary_text)
                    
                    summary_point = PointStruct(
                        id=int(time.time() * 1000000),
                        vector=summary_embedding,
                        payload={
                            "page_content": summary_text,
                            "file_path": file_path,
                            "user_id": user_id,
                            "filename": file_path.split("/")[-1],
                            "ingestion_target": "neo4j"  # Mark as graph database
                        }
                    )
                    
                    # Upsert to document_summaries collection
                    try:
                        qdrant_client.upsert(
                            collection_name="document_summaries",
                            points=[summary_point]
                        )
                        log.info("✓ Summary stored in Qdrant for planner retrieval")
                    except Exception:
                        # Create collection if doesn't exist
                        try:
                            from qdrant_client.models import Distance, VectorParams
                            qdrant_client.recreate_collection(
                                collection_name="document_summaries",
                                vectors_config=VectorParams(
                                    size=len(summary_embedding),
                                    distance=Distance.COSINE
                                )
                            )
                            qdrant_client.upsert(
                                collection_name="document_summaries",
                                points=[summary_point]
                            )
                            log.info("✓ Created document_summaries collection and stored summary")
                        except Exception as e:
                            log.error(f"Failed to create/upsert summary: {e}")
                            
                except Exception as e:
                    log.warning(f"Failed to generate/store summary (non-critical): {e}")

            log.info("✓ Graph ingestion completed successfully")
            return True
            
        except Exception as e:
            log.error(f"Graph ingestion failed: {e}")
            return False
