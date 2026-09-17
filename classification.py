"""
classification.py - ask an LLM to classify Wikipedia edits.

The task
--------
For every edit we ask one question: does this change alter the meaning, framing
or interpretation of the article in a way that could be ideologically
significant, derogatory or manipulative?  The model answers with one of three
labels:

    Potentially Manipulative   the change shifts how readers understand the subject
    Not Problematic            the change does not affect interpretation or framing
    Ambiguous                  the change cannot be judged from text alone (for
                               example an image swap)

This is the same three-label task, the same prompt wording and the same input
layout as the main CROSS2026 pipeline, so results stay comparable.  The code is
a simplified version of ``manipulation_classification/classification.py``.

How a request works
-------------------
Every call is independent: we send a system prompt (the rules) and a user
message (one edit), and we read the label out of the answer.  Sending many
requests in parallel is the whole point of the async code below: 10 edits one
after another take 10 times longer than 10 edits at once.

    from utils import get_client
    from classification import classify_revisions
    from preprocessing import load_revisions, sample_revisions

    edits = sample_revisions(load_revisions("Ukraine", n=1000), n=10, seed=42)
    results = await classify_revisions(edits, client=get_client())
"""

from __future__ import annotations

import asyncio
import re
from asyncio import Semaphore
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI, BadRequestError

from checkpoint import load_checkpoint, save_checkpoint
from preprocessing import build_llm_input, text
from utils import MODEL, get_client

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional, the run works without it
    tqdm = None

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

# The three labels, spelled exactly as the model must spell them.
LABELS = ("Potentially Manipulative", "Not Problematic", "Ambiguous")

# Temperature controls randomness.  0.2 keeps the answers stable: we want the
# same edit to receive the same label twice, otherwise a bias analysis would
# measure noise instead of bias.
DEFAULT_TEMPERATURE = 0.2

# Fixed seed, same reason.  Not every model honours it, so it is a help rather
# than a guarantee.
DEFAULT_SEED = 42

# How many requests are in flight at the same time.  Higher is faster, but the
# server starts refusing requests when the value is too high.
DEFAULT_MAX_CONCURRENT = 5

# Hard deadline for one request, in seconds.  Without it a single stuck request
# can freeze a whole run.
DEFAULT_TIMEOUT = 180.0

# When a diff is longer than this, we cut it and retry once.  This is the
# safety net for the "context window exceeded" error, which simply means the
# text did not fit in the model's memory.
MAX_DIFF_CHARS = 40_000


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------

# The system prompt holds the task definition and the rules.  It is sent with
# every request, and it is the only thing to edit when tuning the classifier.
# Keeping it in one constant makes runs comparable: a change here is a change
# of method, so record it when you report results.
SYSTEM_PROMPT = """\
You are an expert Wikipedia editor analysing revision diffs for cultural manipulation.

## Task
Classify each Wikipedia edit as POTENTIALLY MANIPULATIVE, NOT PROBLEMATIC, or AMBIGUOUS.

## Decision rule
Ask yourself: does this change alter the meaning, framing, interpretation, contextual
understanding, or readability of the text in a way that is derogatory, manipulative or
ideologically significant?  Judge the EFFECT of the change, not the editor's intention.

POTENTIALLY MANIPULATIVE - the change shifts how readers understand, perceive or interpret
the subject. Examples:
- Swapping terms for ideologically loaded alternatives (Kiev to Kyiv, "annexation" to
  "reunification", "genocide" to "alleged genocide")
- Adding or removing text that introduces bias, one-sided emphasis or emotional charge
- Removing inconvenient facts or citations, as opposed to bulk cleanup
- Adding or removing quality or neutrality tags ({{POV}}, {{disputed}}) when used to
  delegitimise content
- Commenting out content with <!-- --> to hide it
- Vandalism or content disruption that degrades article quality and readability
- Toponym changes that systematically impose one naming convention over another, in
  either direction

NOT PROBLEMATIC - the change does not meaningfully affect interpretation or framing. Examples:
- Grammar, spelling, punctuation or formatting fixes
- Neutral factual additions or corrections that do not shift the broader narrative
- Technical or maintenance edits
- Talk page edits (editor discussion, not article content)
- Removing genuinely offensive or biased language, which improves neutrality

AMBIGUOUS - the diff only involves content that cannot be judged from text alone, for
example image or audio changes, or unclear external links. Always classify visual only
edits as AMBIGUOUS.

## Key principle
An edit that is ideologically loaded but factually correct is NOT manipulative. Use your
best judgement.

## Output format
Respond with exactly one label followed by a short justification:
Label: <Potentially Manipulative | Not Problematic | Ambiguous>. [Two sentences explaining your choice]"""


# --------------------------------------------------------------------------
# Reading the answer
# --------------------------------------------------------------------------

# Reasoning models sometimes wrap their thinking in such a block.  We remove it
# so the label is read from the final answer only.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Looks for the first label the model writes.  Older wording ("Weaponised") is
# accepted too, so results from earlier runs stay comparable.
_LABEL_RE = re.compile(
    r"(potentially manipulative|not problematic|ambiguous|weaponised|not weaponised)"
    r"\s*[.:]?\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)

# Canonical spelling used everywhere in this project.
# handling of "weaponised" is for backwards compatibility with earlier runs.
_LABEL_CANONICAL = {
    "potentially manipulative": "Potentially Manipulative",
    "not problematic": "Not Problematic",
    "ambiguous": "Ambiguous",
    "weaponised": "Potentially Manipulative",
    "not weaponised": "Not Problematic",
}


def parse_classification(raw: str) -> tuple[str, str]:
    """Split a model answer into ``(label, reasoning)``.

    The model is asked to reply with "Label: X. explanation".  We take the
    first label it mentions and treat the rest as the justification.

    When nothing matches, the label is ``"Unknown"`` and the whole answer is
    returned as the reasoning.  That case is worth inspecting: it usually means
    the model refused the task or answered in an unexpected format.
    """
    cleaned = _THINK_RE.sub("", raw or "").strip()

    match = _LABEL_RE.search(cleaned)
    if not match:
        return "Unknown", cleaned

    label = _LABEL_CANONICAL[match.group(1).lower()]
    return label, match.group(2).strip()


def _short(value, limit: int = 300) -> str:
    """Shorten a string for display in lists and tables."""
    value = text(value)
    return value if len(value) <= limit else value[:limit] + "..."


# --------------------------------------------------------------------------
# One request
# --------------------------------------------------------------------------


async def classify_single(
    sem: Semaphore,
    client: AsyncOpenAI,
    edit,
    model: str = MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int | None = DEFAULT_SEED,
    timeout: float = DEFAULT_TIMEOUT,
    verbose: bool = False,
) -> dict:
    """Classify one edit and always return a result dict.

    Never raises.  If anything goes wrong the result carries ``label="Error"``
    and the message in ``reasoning``, so a single bad edit cannot kill a long
    run.

    Parameters
    ----------
    sem:
        Semaphore limiting how many requests run at the same time.  It is
        created once by the caller and shared by every task.
    client:
        The async client from ``utils.get_client()``.
    edit:
        One row of the DataFrame produced by ``preprocessing.load_revisions``.
    """
    idx = edit["idx"] if "idx" in edit else -1

    # The metadata is copied into the result, so a result row is readable on
    # its own, without joining it back to the source table.
    result = {
        "idx": idx,
        "page_title": text(edit["page_title"] if "page_title" in edit else ""),
        "section": text(edit["section"] if "section" in edit else ""),
        "comment": text(edit["comment"] if "comment" in edit else ""),
        "label": "Error",
        "reasoning": "",
        "error": "",
        "truncated": False,
        "model": model,
    }

    async with sem:
        # Try the full diff first, then fall back to a cut version.
        for attempt, max_chars in enumerate((None, MAX_DIFF_CHARS)):
            llm_input = build_llm_input(edit, max_chars=max_chars)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": llm_input},
            ]
            try:
                request = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    seed=seed,
                )
                # wait_for adds a deadline of its own on top of the client
                # timeout, because a stuck request would block its slot in the
                # semaphore forever.
                response = await asyncio.wait_for(request, timeout=timeout)
            except (asyncio.TimeoutError, TimeoutError) as exc:
                result["error"] = f"Timeout after {timeout} seconds: {exc}"
                return result
            except BadRequestError as exc:
                # A context window overflow means the diff was too long.  Cut it
                # and try once more; any other bad request is a real failure.
                if "context" not in str(exc).lower() or attempt == 1:
                    result["error"] = f"BadRequestError: {exc}"
                    return result
                if verbose:
                    print(f"  [{idx}] diff too long, retrying with {MAX_DIFF_CHARS} characters")
                continue
            except Exception as exc:  # noqa: BLE001 - keep the run alive
                result["error"] = f"{type(exc).__name__}: {exc}"
                return result

            message = response.choices[0].message
            raw = (message.content or "").strip()
            if not raw:
                # Reasoning models put their thinking in a separate field and
                # can leave the answer empty when the budget ran out.  Falling
                # back on it still beats returning an unusable "Unknown".
                raw = getattr(message, "reasoning_content", "") or ""
            label, reasoning = parse_classification(raw)

            result["label"] = label
            result["reasoning"] = reasoning
            result["truncated"] = max_chars is not None
            if verbose:
                print(f"  [{idx}] {label} - {_short(reasoning, 80)}")
            return result

    # Unreachable, the loop always returns.
    return result


# --------------------------------------------------------------------------
# Many requests
# --------------------------------------------------------------------------


async def classify_stream(
    revisions: pd.DataFrame,
    client: AsyncOpenAI | None = None,
    model: str = MODEL,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int | None = DEFAULT_SEED,
    timeout: float = DEFAULT_TIMEOUT,
    skip_indices: set | None = None,
    show_progress: bool = True,
    verbose: bool = False,
):
    """Classify every edit and yield results as they arrive.

    This is an *async generator*: use it with ``async for`` and each result
    appears as soon as its request comes back, not in the order of the input.

    It is designed for long runs, because the caller decides what to do with
    each result, for example saving it immediately to disk:

        async for result in classify_stream(edits, client, skip_indices=done):
            save_checkpoint(result, "checkpoints/ukraine_qwen.jsonl")

    Parameters
    ----------
    revisions:
        DataFrame of edits, as returned by ``preprocessing.load_revisions``.
    client:
        Async client.  Created automatically when omitted.
    skip_indices:
        Set of ``idx`` values to ignore, normally read from a checkpoint file
        with ``checkpoint.load_checkpoint``.  This is how a run resumes.
    """
    if client is None:
        client = get_client()
    skip_indices = skip_indices or set()

    # The semaphore is what protects the server: only max_concurrent requests
    # are actually in flight, whatever the number of tasks.
    sem = Semaphore(max_concurrent)

    rows = [row for _, row in revisions.iterrows() if row["idx"] not in skip_indices]
    if verbose:
        print(f"Classifying {len(rows)} edits with {model} ({max_concurrent} at a time)")

    # One task per edit.  as_completed gives them back in the order they finish,
    # so the slowest edit never holds up the results of the fast ones.
    tasks = [
        asyncio.create_task(
            classify_single(
                sem,
                client,
                row,
                model=model,
                temperature=temperature,
                seed=seed,
                timeout=timeout,
                verbose=verbose,
            )
        )
        for row in rows
    ]

    progress = tqdm(total=len(tasks), desc="Classifying") if (show_progress and tqdm) else None
    try:
        for finished in asyncio.as_completed(tasks):
            result = await finished
            if progress is not None:
                progress.update(1)
            yield result
    finally:
        if progress is not None:
            progress.close()


async def classify_revisions(
    revisions: pd.DataFrame,
    client: AsyncOpenAI | None = None,
    model: str = MODEL,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int | None = DEFAULT_SEED,
    timeout: float = DEFAULT_TIMEOUT,
    checkpoint_path: str | Path | None = None,
    show_progress: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """Classify a DataFrame and return the results as a list of dicts.

    This is the convenient entry point: it takes care of the client, of
    resuming from a checkpoint file and of saving each result while the run is
    going.

    Parameters
    ----------
    checkpoint_path:
        Where to store results line by line.  Strongly recommended for runs
        longer than a few hundred edits: if the run stops, call the function
        again with the same path and it continues where it left off.

    Returns
    -------
    list[dict]
        One dict per edit, sorted by ``idx``.  The keys are the same as in
        ``classify_single``, so the list is easy to turn into a DataFrame.
    """
    if client is None:
        client = get_client()

    # Load whatever a previous run already finished.
    previous: list[dict] = []
    done: set = set()
    if checkpoint_path is not None:
        previous, done = load_checkpoint(checkpoint_path)
        if previous and verbose:
            print(f"Resuming: {len(previous)} edits already in {checkpoint_path}")

    new_results: list[dict] = []
    async for result in classify_stream(
        revisions,
        client=client,
        model=model,
        max_concurrent=max_concurrent,
        temperature=temperature,
        seed=seed,
        timeout=timeout,
        skip_indices=done,
        show_progress=show_progress,
        verbose=verbose,
    ):
        new_results.append(result)
        if checkpoint_path is not None:
            # Save straight away: at this point the result is only in memory.
            save_checkpoint(result, checkpoint_path)

    results = previous + new_results
    results.sort(key=lambda record: record.get("idx", -1))
    return results


# --------------------------------------------------------------------------
# Presenting results
# --------------------------------------------------------------------------


def results_to_dataframe(
    results: list[dict],
    revisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turn a list of result dicts into a DataFrame.

    Parameters
    ----------
    results:
        Output of ``classify_revisions``.
    revisions:
        The source DataFrame.  When given, the diff, the date and the editor
        are joined back onto each result by ``idx``, which is what you want
        before saving or inspecting the run.
    """
    df = pd.DataFrame(results)
    if df.empty:
        return df

    if revisions is not None:
        columns = [name for name in ("idx", "changed_version", "date", "user") if name in revisions.columns]
        df = df.merge(revisions[columns], on="idx", how="left")
    return df


def save_results(
    results: list[dict],
    path: str | Path,
    revisions: pd.DataFrame | None = None,
) -> Path:
    """Write the results to a CSV file and return its path.

    ``encoding="utf-8-sig"`` adds a byte order mark, which makes Excel open
    files containing Cyrillic or Armenian text without garbling them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = results_to_dataframe(results, revisions)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
