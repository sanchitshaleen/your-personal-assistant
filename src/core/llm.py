"""LLM system module for managing language model interactions.
This module provides functions to initialize and manage LLM models, and Parsers
Also contains dummy response generators for testing purposes.
"""

from time import sleep
from random import choice
from typing import Generator

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel as T_LLM

from src.core.logger import get_logger
log = get_logger(name="core_llm")


def get_llm(model_name: str, context_size: int,
            temperature: float, verify_connection: bool = False) -> T_LLM:
    """Get the LLM model with the specified parameters.

    Args:
        model_name (str): The name of the LLM model to use.
        context_size (int): The maximum context size for the model.
        temperature (float): The temperature setting for the model.
        verify_connection (bool): Whether to verify the connection to the model.

    Returns:
        BaseChatModel: An instance of the LLM model configured with the specified parameters.
    """

    log.info(f"Initializing LLM(model={model_name}, ctx_size={context_size}, temp={temperature})")
    model = ChatOllama(
        model=model_name, 
        num_ctx=8192,  # Safe context window for CPU inference
        temperature=temperature, 
        num_predict=1000,  # Allow comprehensive responses (~2000 words)
        keep_alive=-1,
        base_url="http://host.docker.internal:11434"
    )

    if verify_connection:
        try:
            _ = model.invoke("ping")
            log.info(f"LLM model '{model_name}' initialized and connection verified.")

        except Exception as e:
            log.error(f"Failed to initialize LLM model '{model_name}': {e}")
            raise RuntimeError(f"Could not initialize LLM model '{model_name}'") from e
    else:
        log.warning(f"LLM model '{model_name}' initialized without connection verification.")

    return model


def get_output_parser():
    """Get the output parser for the LLM model.

    Returns:
        StrOutputParser: An instance of StrOutputParser to parse the model's output.
    """
    log.info("Initializing the output parser for LLM responses.")
    return StrOutputParser()


# ------------------------------------------------------------------------------
# Dummy responses of LLM
# ------------------------------------------------------------------------------

# Resp in increasing order of length:
dummy_responses = [
    "1> Hey there! 👋  \nI'm Gemma-3, a language model developed by the Gemma team at Google. I'm here to help you with questions and create interesting text. I’m still improving and learning each day. Let’s explore and have some fun together! 🤖",

    "2> Hello 👋!  \nI'm Gemma-3 😎, a large language model created by the Gemma team at Google-Deepmind. I’m here to assist you with a wide range of tasks, from answering your questions to generating creative text formats. My goal is to provide helpful and informative responses. I'm still under development, and I’m learning new things every day! I’m excited to explore with you. Let's see what we can create! 🤖✨",

    "3> Hello 👋!  \nI'm Gemma-3 😎, a powerful language model developed by the talented folks at Google-Deepmind. I'm here to help you out with a wide range of tasks—whether it’s answering complex questions, crafting detailed explanations, writing stories, poems, or even generating code snippets. I strive to be informative, creative, and engaging in every response I give.  \n\nI'm constantly learning, improving, and adapting to serve you better. Even though I'm still a work in progress, I'm pretty good at what I do! 😄  \n\nFeel free to test my capabilities—ask me anything, challenge me, or just chat. Let’s collaborate, learn new things, and build something awesome together. Ready when you are! 🚀🤖✨"
]
log.info(f"Loaded {len(dummy_responses)} dummy responses for testing.")


def get_dummy_response() -> str:
    """Get a dummy response from the predefined list."""
    return choice(dummy_responses)


def get_dummy_response_stream(batch_tokens: int, token_rate: int = 45) -> Generator[str, None, None]:
    """Get a dummy response from the predefined list."""
    delay = batch_tokens / token_rate
    dummy_response = choice(dummy_responses)

    for i in range(0, len(dummy_response.split(" ")), batch_tokens):
        yield " ".join(dummy_response.split(" ")[i:i + batch_tokens])+" "
        sleep(delay)
