"""
utils.py - shared helpers for talking to the LLM API.

Every script in this project sends requests to the EPFL RCP inference server.
That server implements the OpenAI HTTP API, which means we can use the official
``openai`` Python package instead of writing HTTP calls by hand.  This is the
same setup as the main CROSS2026 pipeline.

Two helpers live here:

``load_api_key()``
    Finds the secret key that authorises our requests.  We look in two places,
    in this order:

    1. the environment variable ``RCP_API_KEY`` (preferred: the secret stays
       out of the repository),
    2. the file ``.github/api_key`` at the repository root (fallback, handy
       while working locally).

``get_client()``
    Builds the client object.  We use the *asynchronous* flavour because we
    later send many requests in parallel (see ``classification.py``).

Quick check inside a notebook:

    from utils import get_client, MODEL

    client = get_client()
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    print(response.choices[0].message.content)
"""

from __future__ import annotations

import os
from pathlib import Path

from openai import AsyncOpenAI

# Base URL of the EPFL RCP inference server.  It speaks the OpenAI API, which
# is why the official openai package works without any adaptation.
RCP_BASE_URL = "https://inference.rcp.epfl.ch/v1"

# Default model for this project.
# Qwen is a good starting point: it is fast on this endpoint and follows the
# classification instructions reliably.  Swapping this single constant is
# enough to test another model, which is exactly what a bias analysis needs.
MODEL = "Qwen/Qwen3.6-35B-A3B"

# Fallback location of the API key file, relative to the repository root.
# The main CROSS2026 repository stores the key in ``.github/api_key``.
_KEY_FILE_RELATIVE_PATH = Path(".github") / "api_key"


def load_api_key(env_var: str = "RCP_API_KEY") -> str:
    """Return the API key, from the environment or from the key file.

    Parameters
    ----------
    env_var:
        Name of the environment variable to read first.

    Returns
    -------
    str
        The API key, without surrounding whitespace.

    Raises
    ------
    RuntimeError
        If neither source provides a key.  The message explains how to fix it,
        because a missing key is the most common setup problem.
    """
    # Option 1: environment variable.  This is the recommended way because the
    # secret never touches the disk or the git history.
    key = os.getenv(env_var)
    if key and key.strip():
        return key.strip()

    # Option 2: key file.  This project folder sits next to the main CROSS2026
    # repository, so ``parents[1]`` is the repository root.
    repo_root = Path(__file__).resolve().parents[1]
    key_file = repo_root / _KEY_FILE_RELATIVE_PATH
    if key_file.exists():
        file_key = key_file.read_text(encoding="utf-8").strip()
        if file_key:
            return file_key

    raise RuntimeError(
        "No API key found.\n"
        f"Either set the environment variable {env_var}, for example:\n"
        f"    export {env_var}=your-key-here\n"
        f"or create the file {key_file} containing only the key."
    )


def get_client(env_var: str = "RCP_API_KEY") -> AsyncOpenAI:
    """Build an asynchronous client pointed at the RCP inference endpoint.

    The client is created once and then reused for every request.  Creating a
    new client per request would open a new HTTP connection pool each time and
    slow the run down for no benefit.
    """
    return AsyncOpenAI(base_url=RCP_BASE_URL, api_key=load_api_key(env_var))


async def check_connection(client: AsyncOpenAI, model: str = MODEL) -> str:
    """Send one tiny request and return the model's answer.

    Useful as a first cell in a notebook: if this works, the key, the network
    and the model name are all correct.

    Usage::

        from utils import check_connection, get_client

        print(await check_connection(get_client()))

    Note that the function is declared with ``async def``.  Calling it returns
    a coroutine, and the ``await`` keyword runs it and gives back the answer.
    """
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()
