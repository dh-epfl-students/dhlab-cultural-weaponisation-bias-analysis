"""
checkpoint.py - crash-safe checkpointing for long classification runs.

Why this file exists
--------------------
A classification run over thousands of edits can take a long time, and a single
network hiccup or a closed laptop should never cost you the whole run.  The
trick is simple: after every finished edit, append its result to a JSONL file
on disk.  If the run stops, restart it and it skips whatever is already there.

JSONL means "one JSON object per line".  It is used because:

* appending one line is atomic, so a crash can only damage the last line,
* the file stays readable with any text editor,
* a partially written line can simply be skipped when reading back.

The main CROSS2026 project uses exactly the same pattern for its multi-hour LLM
runs, which is why the function names match: ``save_checkpoint`` and
``load_checkpoint``.

Typical use
-----------
    from checkpoint import load_checkpoint, save_checkpoint
    from classification import classify_stream

    path = "checkpoints/ukraine_qwen.jsonl"

    # 1. Look up what was already done, and skip those rows.
    done_results, done_indices = load_checkpoint(path)
    print(f"{len(done_indices)} edits already classified")

    # 2. Classify only the missing rows, saving each result as it arrives.
    async for result in classify_stream(revisions, client, skip_indices=done_indices):
        done_results.append(result)
        save_checkpoint(result, path)
"""

from __future__ import annotations

import json
from pathlib import Path

# The column used to recognise an already-processed edit.  Every result dict
# produced by classification.py contains this key.
DEFAULT_KEY_FIELD = "idx"


def save_checkpoint(result: dict, path: str | Path) -> None:
    """Append one result to the checkpoint file.

    The parent folder is created if needed, so you can pass any path you like.
    ``ensure_ascii=False`` keeps the original characters (Cyrillic, Armenian
    script, accents) readable in the file instead of escaping them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_checkpoint(
    path: str | Path,
    key_field: str = DEFAULT_KEY_FIELD,
) -> tuple[list[dict], set]:
    """Read a checkpoint file back into memory.

    Parameters
    ----------
    path:
        Path to the JSONL file.  A missing file is not an error: it simply
        means nothing has been done yet, so the function returns empty values.
    key_field:
        Field used to build the set of finished keys, "idx" by default.

    Returns
    -------
    (results, done_keys)
        ``results`` is the list of all stored result dicts and ``done_keys``
        the set of values found under ``key_field``.  Pass ``done_keys`` as the
        ``skip_indices`` argument of ``classify_stream`` to resume.
    """
    path = Path(path)
    if not path.exists():
        return [], set()

    results: list[dict] = []
    done_keys: set = set()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A damaged line can only be the last one (writes are atomic),
                # so it is safe to ignore it and keep the earlier results.
                continue
            results.append(record)
            if key_field in record:
                done_keys.add(record[key_field])

    return results, done_keys


def checkpoint_summary(path: str | Path, key_field: str = DEFAULT_KEY_FIELD) -> dict:
    """Return a small description of a checkpoint file.

    Handy in a notebook to answer "where did my last run stop?".
    """
    path = Path(path)
    results, done_keys = load_checkpoint(path, key_field=key_field)
    labels: dict[str, int] = {}
    for record in results:
        label = record.get("label", "?")
        labels[label] = labels.get(label, 0) + 1
    return {
        "path": str(path),
        "exists": path.exists(),
        "n_results": len(results),
        "n_unique_keys": len(done_keys),
        "labels": labels,
    }
