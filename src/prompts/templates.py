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
            "You are a direct and concise document assistant.\n"
            "\n"
            "CONTEXT FROM DOCUMENTS:\n{context}\n"
            "\n"
            "ANSWERING RULES:\n"
            "1. If the user asks about PREVIOUS QUESTIONS or CONVERSATION HISTORY (e.g., 'what did I ask', 'my last question', 'what was my earlier question'):\n"
            "   - LOOK AT THE 'HISTORY' SECTION BELOW.\n"
            "   - Find the last 'HumanMessage' in the list.\n"
            "   - Answer clearly: 'You asked about [topic]...'\n"
            "   - Do NOT answer the previous question again, just describe what it was.\n"
            "\n"
            "2. If the user asks about information they provided in this conversation (e.g., their name, preferences):\n"
            "   - Answer using the 'History' section.\n"
            "   - CRITICAL: Even if 'Context from Documents' is empty or says 'No documents found', you MUST use the History to answer personal questions.\n"
            "\n"
            "3. For ALL OTHER questions (document queries):\n"
            "   - Answer primarily using the provided 'Context from Documents'\n"
            "   - If the answer is in 'History' but not in 'Documents', you MAY use 'History' if it provides continuity\n"
            "   - If the 'Context from Documents' does not contain the answer and it's not in History, say 'I cannot find the answer in the documents.'\n"
            "\n"
            "FORMATTING:\n"
            "- Be direct. Do not start with 'Okay', 'Sure', 'Here is', or 'Based on'\n"
            "- Do not include filenames or source metadata in responses\n"
        )),
        ("system", "History:\n"),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "End of History.\n"),
        ("human", "Question: {input}\n\n(Reminder: If asking about history, ignore empty context and answer from History above.)")
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
