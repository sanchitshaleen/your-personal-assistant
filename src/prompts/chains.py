from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableWithMessageHistory, Runnable
from langchain_core.documents import Document

from .templates import template_chat as chat_prompt
from .templates import template_summarize as summary_prompt

from typing import Callable, List, Any
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.logger import get_logger
log = get_logger(name="chains_rag")


class HybridRetrieverWrapper(Runnable):
    """Wrapper to make HybridRetriever compatible with LangChain chains."""
    
    def __init__(self, hybrid_retriever):
        self.hybrid_retriever = hybrid_retriever
        self.selected_files = []  # Will be set by config
    
    def invoke(self, input_data: str, config=None, **kwargs) -> List[Document]:
        """Retrieve documents using hybrid retrieval with optional file filtering."""
        if isinstance(input_data, dict):
            query = input_data.get('input', str(input_data))
        else:
            query = str(input_data)
        
        # Extract selected files from config if available
        selected_files = []
        if config and isinstance(config, dict):
            selected_files = config.get('configurable', {}).get('selected_files', [])
        
        log.info(f"HybridRetrieverWrapper.invoke called with query: '{query[:80]}'")
        log.info(f"Selected files filter: {selected_files if selected_files else 'None (using all files)'}")
        
        docs = self.hybrid_retriever.retrieve_hybrid(query, selected_files=selected_files)
        log.info(f"HybridRetrieverWrapper returned {len(docs)} documents")
        for i, doc in enumerate(docs[:3]):
            filename = doc.metadata.get('filename', 'unknown')
            log.info(f"  Doc {i+1} (from {filename}): {doc.page_content[:100]}...")
        return docs
    
    def batch(self, inputs: List[str], config=None, **kwargs) -> List[List[Document]]:
        """Batch retrieval."""
        return [self.invoke(inp, config) for inp in inputs]
    
    def stream(self, input_data: str, config=None, **kwargs):
        """Stream retrieval results."""
        yield self.invoke(input_data, config)



def build_rag_chain(
        llm_chat: BaseChatModel, llm_summary: BaseChatModel,
        retriever: VectorStoreRetriever, get_history_fn: Callable,
        use_hybrid: bool = False):
    """Builds a Conversational RAG (Retrieval-Augmented Generation) chain.

    Args:
        llm_chat (BaseChatModel): The LLM model for generating chat responses.
        llm_summary (BaseChatModel): The LLM model for summarizing chat history.
        retriever (VectorStoreRetriever): The retriever to fetch relevant documents.
        get_history_fn (Callable): Function to retrieve chat history for a session.
        use_hybrid (bool): Whether to use hybrid retrieval (BM25 + Semantic + Qwen3). Defaults to False.

    Returns:
        RunnableWithMessageHistory: A runnable chain that processes user input and chat history
        to provide a final answer based on retrieved documents and chat context.
    """

    log.info(f"Building the Conversational RAG Chain (hybrid={use_hybrid})...")

    # If using hybrid retriever, wrap it
    if use_hybrid and hasattr(retriever, 'retrieve_hybrid'):
        log.info("Using HybridRetrieverWrapper for advanced retrieval")
        retriever = HybridRetrieverWrapper(retriever)

    # Chain to summarize the history and retrieve relevant documents
    # 3 User Input + Chat History > Summarizer Template > Standalone Que > Get Docs
    retriever_chain = create_history_aware_retriever(llm_summary, retriever, summary_prompt)
    log.info("Created the retriever chain with summarization.")
    log.info(f"Retriever chain type: {type(retriever_chain)}")

    # Chain to combine the retrieved documents and get the final answer
    # Use a simple manual approach without create_stuff_documents_chain complications
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    
    def format_docs_for_chain(input_dict):
        """Format documents and pass through to prompt, while tracking sources."""
        try:
            docs = input_dict.get("context", [])
            log.info(f"format_docs_for_chain: Got {len(docs)} docs, type: {type(docs)}")
            
            # Handle case where docs might be strings or Document objects
            formatted_docs = []
            sources = set()
            
            for i, doc in enumerate(docs):
                try:
                    if hasattr(doc, 'page_content'):
                        formatted_docs.append(doc)
                        if hasattr(doc, 'metadata') and doc.metadata:
                            if 'filename' in doc.metadata:
                                sources.add(doc.metadata['filename'])
                            elif 'source' in doc.metadata:
                                sources.add(doc.metadata['source'])
                    elif isinstance(doc, str):
                        # If doc is a string, convert to Document
                        from langchain_core.documents import Document
                        formatted_docs.append(Document(page_content=doc))
                        sources.add("Unknown")
                    else:
                        log.warning(f"format_docs_for_chain: Doc {i} has unexpected type: {type(doc)}")
                except Exception as e:
                    log.error(f"format_docs_for_chain: Error processing doc {i}: {e}")
            
            # Build context string from formatted documents, include filename/source explicitly
            context_entries = []
            for i, doc in enumerate(formatted_docs[:3]):
                src = getattr(doc, 'metadata', {}) or {}
                filename = src.get('filename') or src.get('source') or f'unknown_{i+1}'
                content_preview = doc.page_content[:500] if hasattr(doc, 'page_content') else str(doc)[:500]
                entry = f"Source: {filename}\nContent: {content_preview}"
                context_entries.append(entry)
            context_str = "\n\n".join(context_entries)
            
            log.info(f"format_docs_for_chain: Formatted {len(formatted_docs)} docs from {len(sources)} sources")
            
            return {
                "context": context_str,
                "chat_history": input_dict.get("chat_history", []),
                "input": input_dict.get("input", ""),
                "sources": list(sources)
            }
        except Exception as e:
            log.error(f"format_docs_for_chain: Critical error: {e}", exc_info=True)
            return {
                "context": "",
                "chat_history": input_dict.get("chat_history", []) if isinstance(input_dict, dict) else [],
                "input": input_dict.get("input", "") if isinstance(input_dict, dict) else "",
                "sources": []
            }
    
    def extract_answer(response):
        """Extract answer text from LLM response."""
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    
    qa_chain = (
        RunnableLambda(format_docs_for_chain)
        | chat_prompt
        | llm_chat
        | RunnableLambda(extract_answer)
    )
    log.info("Created the simple QA chain.")

    # Main RAG Chain:
    # 2 Input + Chat History > [ `Summarizer Template` > `Get Docs` ] > [ `Combine` > `Chat Template` ] > Output
    rag_chain = create_retrieval_chain(retriever_chain, qa_chain)
    log.info("Created the main RAG chain.")
    log.info(f"RAG chain type: {type(rag_chain)}")

    log.info("Returning the final Conversational RAG Chain w history.")
    # 1 Final Conversational RAG Chain:
    return RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_history_fn,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
