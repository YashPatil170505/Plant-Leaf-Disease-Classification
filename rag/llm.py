"""
llm.py - Thin wrapper around the Gemini API (google-genai SDK)

Gemini is used ONLY to produce the conversational turns (deciding which local
tool to call, then writing the answer). Prediction, embeddings, retrieval, tool
execution, report generation and evaluation scoring all run locally.

Reads GOOGLE_API_KEY from the environment / .env file. Free-tier models are
often overloaded, so each call has a timeout; on overload/timeout we move on to
the next model in GEMINI_MODELS instead of waiting, and on rate limits (429) we
back off before retrying.
"""

import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

load_dotenv()

CHAT_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3.8-flash,gemini-3.5-flash,gemini-3.1-flash-lite").split(",")]
TIMEOUT_S   = float(os.getenv("GEMINI_TIMEOUT_S", "60"))
OVERLOADED  = {500, 503, 504}

_client = None


def client():
    global _client
    if _client is None:
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is not set. Put it in a .env file in the project root.")
        _client = genai.Client(
            api_key=os.environ["GOOGLE_API_KEY"],
            http_options=types.HttpOptions(
                timeout=int(TIMEOUT_S * 1000),                       # milliseconds
                retry_options=types.HttpRetryOptions(attempts=1),    # we handle retries/fallback here
            ),
        )
    return _client


def chat_turn(contents, system, tools, rate_limit_retries=3):
    """One model call over the running conversation. Returns (response, model_used).

    Automatic function calling is disabled: the caller executes tool calls itself
    (locally) and appends the results, so every tool call is visible and logged.
    """
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    last_err = None
    for model in CHAT_MODELS:
        for attempt in range(rate_limit_retries):
            try:
                return client().models.generate_content(model=model, contents=contents, config=config), model
            except errors.APIError as e:
                last_err = e
                if e.code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                if e.code in OVERLOADED:
                    break                       # try the next model right away
                raise
            except Exception as e:              # timeouts / connection errors
                last_err = e
                break
    raise RuntimeError(f"All Gemini models failed: {last_err}")
