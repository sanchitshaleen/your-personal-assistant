from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import redis
import json
import os
import time

from src.core.logger import get_logger

log = get_logger(name="core_history")

# Redis connection parameters
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = 0

# Get Redis client with retry logic
def get_redis_client(max_retries=5, retry_delay=2):
    """Initialize Redis client with retry logic for resilience.
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        redis.Redis client or None if connection fails after all retries
    """
    for attempt in range(max_retries):
        try:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30
            )
            client.ping()
            log.info(f"✓ Connected to Redis at {REDIS_HOST}:{REDIS_PORT} (attempt {attempt + 1})")
            return client
        except redis.ConnectionError as e:
            if attempt < max_retries - 1:
                log.warning(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                log.error(f"✗ Failed to connect to Redis after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            log.error(f"Unexpected error connecting to Redis: {e}")
            return None
    return None

redis_client = get_redis_client()


class RedisChatHistory(BaseChatMessageHistory):
    """A Redis-backed chat history store for persistent message storage.
    
    This class implements BaseChatMessageHistory to provide persistent message storage
    using Redis as the backend. Messages are serialized as JSON and stored with a key
    pattern: `chat_history:{session_id}`. This ensures chat history survives server restarts.

    Attributes:
        session_id (str): The unique identifier for the chat session (typically user_id).
        messages (list): In-memory cache of messages for the current session.
    """

    def __init__(self, session_id: str):
        """Initialize the chat history for a specific session.

        Args:
            session_id (str): The unique identifier for the chat session (user_id).
        """
        self.session_id = session_id
        self.redis_key = f"chat_history:{session_id}"
        self.messages = []
        self._load_from_redis()
        log.info(f"Initialized RedisChatHistory for session '{session_id}' with {len(self.messages)} messages")

    def _load_from_redis(self):
        """Load chat messages from Redis into memory."""
        if redis_client is None:
            log.warning(f"Redis not available, starting with empty history for session '{self.session_id}'")
            return

        try:
            # Get all messages from Redis list
            messages_json = redis_client.lrange(self.redis_key, 0, -1)
            self.messages = []
            
            for msg_json in messages_json:
                try:
                    msg_data = json.loads(msg_json)
                    if msg_data.get('role') == 'human':
                        self.messages.append(HumanMessage(content=msg_data.get('content', '')))
                    elif msg_data.get('role') == 'assistant':
                        self.messages.append(AIMessage(content=msg_data.get('content', '')))
                except json.JSONDecodeError as e:
                    log.warning(f"Failed to parse message from Redis: {e}")
            
            log.info(f"Loaded {len(self.messages)} messages from Redis for session '{self.session_id}'")
        except Exception as e:
            log.error(f"Error loading messages from Redis: {e}")
            self.messages = []

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat history (both in-memory and persistent Redis storage).

        Args:
            message (BaseMessage): The message to add (HumanMessage or AIMessage).
        """
        self.messages.append(message)
        
        if redis_client is None:
            log.warning(f"Redis not available, message not persisted for session '{self.session_id}'")
            return
        
        # Determine the role based on message type
        role = 'human' if isinstance(message, HumanMessage) else 'assistant'
        content = message.content if hasattr(message, 'content') else str(message)
        
        # Serialize and persist to Redis
        try:
            msg_data = json.dumps({
                'role': role,
                'content': message.content
            })
            redis_client.rpush(self.redis_key, msg_data)
            log.info(f"Message persisted to Redis for session '{self.session_id}'")
        except Exception as e:
            log.error(f"Failed to persist message to Redis for session '{self.session_id}': {e}")

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add multiple messages to the chat history.

        Args:
            messages (list[BaseMessage]): List of messages to add.
        """
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        """Clear all messages from the chat history (both in-memory and Redis storage)."""
        self.messages = []
        
        if redis_client is None:
            log.warning(f"Redis not available, in-memory history cleared for session '{self.session_id}'")
            return
        
        try:
            redis_client.delete(self.redis_key)
            log.info(f"Cleared chat history from Redis for session '{self.session_id}'")
        except Exception as e:
            log.error(f"Failed to clear chat history from Redis for session '{self.session_id}': {e}")

    @property
    def messages_list(self) -> list[BaseMessage]:
        """Get the list of messages in the current session.

        Returns:
            list[BaseMessage]: All messages in the session.
        """
        return self.messages


class HistoryStore:
    """A manager class to handle multiple chat sessions with persistent memory.

    This class maintains a mapping of session IDs to their respective chat histories,
    using RedisChatHistory for each session. Histories are cached in memory
    during a session but backed by Redis for persistence across server restarts.

    Functions:
        + `get_session_history(session_id: str)`: Retrieves or creates the chat history for a given session ID.
        + `clear_session_history(session_id: str)`: Clears the chat history for a given session ID.
    """

    def __init__(self):
        """Initialize the history store with an empty session dictionary."""
        self.histories = {}
        log.info("Initialized HistoryStore with persistent Redis backend.")

    def get_session_history(self, session_id: str) -> RedisChatHistory:
        """Retrieves the chat history for a given session ID.
        
        If no history exists in the in-memory cache, it creates a new one and loads
        messages from Redis if they exist.

        Args:
            session_id (str): The unique identifier for the chat session.
            
        Returns:
            RedisChatHistory: The persistent chat message history for the session.
        """
        log.info(f"[DEBUG HISTORY] get_session_history called for session_id: '{session_id}'")
        
        if session_id not in self.histories:
            self.histories[session_id] = RedisChatHistory(session_id)
            log.info(f"Created/loaded history for session: `{session_id}` with {len(self.histories[session_id].messages)} messages")
        else:
            log.info(f"Retrieved cached history for session: `{session_id}` with {len(self.histories[session_id].messages)} messages")

        log.info(f"[DEBUG HISTORY] Returning history with {len(self.histories[session_id].messages)} messages")
        return self.histories[session_id]

    def clear_session_history(self, session_id: str) -> bool:
        """Clears the chat history for a given session ID.

        Args:
            session_id (str): The unique identifier for the chat session.
            
        Returns:
            bool: True if the history was cleared, False if no history existed for the session.
        """
        if session_id in self.histories:
            self.histories[session_id].clear()
            del self.histories[session_id]
            log.info(f"Cleared history for session: `{session_id}`")
            return True
        else:
            # Even if not in cache, still try to clear from Redis
            if redis_client is not None:
                try:
                    redis_client.delete(f"chat_history:{session_id}")
                    log.info(f"Cleared Redis history for session: `{session_id}` (not in cache)")
                    return True
                except Exception as e:
                    log.error(f"Failed to clear Redis history for session: `{session_id}`: {e}")
            
            log.warning(f"No history found for session: `{session_id}` to clear.")
            return False

    def add_interaction(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """Add a user-assistant interaction to the session history.

        Args:
            session_id (str): The unique identifier for the chat session.
            user_message (str): The user's message.
            assistant_message (str): The assistant's response.
        """
        history = self.get_session_history(session_id)
        history.add_message(HumanMessage(content=user_message))
        history.add_message(AIMessage(content=assistant_message))
        log.info(f"Added interaction to session: `{session_id}`")

    def get_history(self, session_id: str) -> list:
        """Get the raw message list for a session.

        Args:
            session_id (str): The unique identifier for the chat session.
            
        Returns:
            list: List of messages (as BaseMessage objects).
        """
        history = self.get_session_history(session_id)
        return history.messages
