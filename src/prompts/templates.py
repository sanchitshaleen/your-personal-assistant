from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

from src.core.logger import get_logger
log = get_logger(name="chains_prompts")

# Simple context string for document formatting
context_prompt = PromptTemplate(
    input_variables=["context"],
    template="Context:\n{context}"
)

# Chat Template - This will be used with create_stuff_documents_chain
template_chat = ChatPromptTemplate.from_messages(
    messages=[
        ("system",  (
            "You are a document assistant. You must answer ONLY using the provided context from the user's uploaded documents.\n"
            "If the answer is not present in the context, or if the question is about general knowledge, real-time information, or anything outside the provided documents, politely respond: 'Sorry, I can only answer questions based on the documents you have uploaded.'\n"
            "Do NOT use your own training knowledge, do NOT make up answers, and do NOT provide general information unless it is explicitly found in the context.\n"
            "If the answer is in the context, provide it directly without disclaimers.\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Focus on answering the SPECIFIC question being asked.\n"
            "2. When multiple context chunks are provided, prioritize chunks that DIRECTLY answer the user's question.\n"
            "3. Extract information that is most relevant and specific to what was asked.\n"
            "4. Avoid including tangential or supplementary information unless the user specifically asks for it.\n"
            "5. If a context chunk contains related information, acknowledge you see it but focus on answering the primary question.\n"
            "6. Be precise: extract exactly what answers the user's question, not broader related information."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Context:\n{context}")  # Add context as a separate message
    ]
)


# Summarizer Template:
template_summarize = ChatPromptTemplate.from_messages(
    messages=[
        ("system", "".join([
            "You are an expert at summarizing conversations into standalone prompts.\n"
            "You are given a complete chat history, ending with the user's latest message.\n\n"
            "SPECIAL CASE - If the user is asking about PREVIOUS MESSAGES or CONVERSATION HISTORY:\n"
            "- Detect keywords like: 'my last question', 'what did I ask', 'my first question', 'previous message', etc.\n"
            "- If detected, return EXACTLY what they asked without modification\n"
            "- Do NOT reformulate questions about the conversation itself\n\n"
            "NORMAL CASE - For all other questions:\n"
            "- Understand the entire conversation context.\n"
            "- Identify references in the latest user message that relate to earlier messages.\n"
            "- Create a single clear, concise, and standalone question or prompt.\n"
            "- This final prompt should be fully understandable without needing the prior conversation.\n"
            "- It will be used to retrieve relevant documents.\n\n"
            "EXAMPLES:\n"
            "- User: 'what was my last question?' → Return: 'what was my last question?'\n"
            "- User: 'tell me more about it' (referring to habits from earlier) → Return: 'Tell me more about the good workplace habits'\n"
            "- User: 'my first question' → Return: 'my first question'\n\n"
            "Only return the rewritten standalone prompt (or original if it's about conversation history). No explanations."
        ])),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

log.info("Initialized chat and summarize prompt templates.")
