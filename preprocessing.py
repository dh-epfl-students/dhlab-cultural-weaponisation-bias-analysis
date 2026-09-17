"""
preprocessing.py - load Wikipedia edits and turn them into LLM input.

This module answers two questions:

1. **Where does the data come from?** ``load_revisions("Ukraine")`` returns a
   pandas DataFrame with one row per edit.
2. **How do we present an edit to the model?** ``build_llm_input(edit)``
   produces the exact text block sent to the LLM, with the editor comment, the
   article section and the page title placed above the raw diff.

This is the same input layout as the main CROSS2026 pipeline, so results from
this project stay comparable with the main one.

The data
--------
Each country has one CSV file with these columns:

    page_title      title of the Wikipedia article
    section         section of the article that was edited (often a heading)
    date            timestamp of the revision
    changed_version the edit itself, stored as a unified diff
    user            username or IP of the editor
    comment         the edit summary written by the editor
    size            size of the article after the edit, in bytes
    tags            edit tags added by Wikipedia (possible vandalism, etc.)
    revid           revision id
    parentid        id of the previous revision

A unified diff looks like this::

    --- old revision
    +++ new revision
    @@ -12,7 +12,7 @@
     unchanged context line
    -text that was removed
    +text that was added

Lines starting with ``-`` were removed, lines starting with ``+`` were added and
lines starting with a space are context.  This project sends the raw diff to the
model, without any markup cleaning, exactly like the current main pipeline.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import pandas as pd

# Python's CSV reader refuses any field longer than 128 KB by default.  A few
# Wikipedia edits are massive rewrites that go well over that, so the limit is
# raised here.  Without this line, loading a country file fails with
# "_csv.Error: field larger than field limit".
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# --------------------------------------------------------------------------
# Where the data lives
# --------------------------------------------------------------------------

# Folder of this file, used to build absolute paths that work from any cwd.
PROJECT_DIR = Path(__file__).resolve().parent

# Official location of the dataset.  Drop the CSVs in ``data/`` and this module
# finds them, no configuration needed.  The raw files are about 2.7 GB per
# country, so they are not shipped with the repository: see the README for
# where to get them.
DATA_DIR = PROJECT_DIR / "data"

# Countries covered by the study.
COUNTRIES = ("Ukraine", "Armenia")

# Columns without which an edit cannot be classified.
REQUIRED_COLUMNS = ("page_title", "section", "changed_version")


def data_file_path(country: str) -> Path:
    """Return the CSV file holding every edit of ``country``.

    Two layouts are accepted inside ``data/``, so you can organise the folder
    the way you prefer::

        data/Ukraine_raw.csv
        data/Ukraine/Ukraine_raw.csv

    Raises
    ------
    ValueError
        If the country is unknown.
    FileNotFoundError
        If no file is found, with a message listing where we looked.
    """
    if country not in COUNTRIES:
        raise ValueError(f"Unknown country '{country}'. Expected one of {COUNTRIES}.")

    filename = f"{country}_raw.csv"

    candidates = [
        DATA_DIR / filename,                 # data/Ukraine_raw.csv
        DATA_DIR / country / filename,       # data/Ukraine/Ukraine_raw.csv
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"No data file found for {country}. I looked in:\n{tried}\n"
        f"Place the CSV in {DATA_DIR} to fix this."
    )


def available_countries() -> dict[str, Path | None]:
    """Return the countries that can currently be loaded.

    The values are the resolved file paths, or ``None`` when the file is
    missing.  This makes the first notebook cell a good sanity check:

        from preprocessing import available_countries
        available_countries()
    """
    found: dict[str, Path | None] = {}
    for country in COUNTRIES:
        try:
            found[country] = data_file_path(country)
        except FileNotFoundError:
            found[country] = None
    return found


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_revisions(
    country: str,
    n: int | None = None,
    seed: int = 42,
    random_sample: bool = True,
) -> pd.DataFrame:
    """Load edits for one country into a DataFrame.

    Parameters
    ----------
    country:
        "Ukraine" or "Armenia" (exact spelling, see ``COUNTRIES``).
    n:
        How many edits to return.  ``None`` loads everything.
        Be careful: a full country file holds several million edits and needs a
        lot of memory, so prefer ``n`` while exploring.
    seed:
        Seed of the random draw, so a run can be reproduced exactly.
    random_sample:
        If True (default), draw ``n`` edits at random from the whole file.
        If False, simply take the first ``n`` edits, which is instant but not
        representative (the file is ordered by article, not shuffled).

    Returns
    -------
    pandas.DataFrame
        The edits, plus an ``idx`` column that identifies each edit.  Text
        columns are stripped and rows without a diff are dropped.

    Notes
    -----
    The random draw uses *reservoir sampling*: the file is read once, line by
    line, and each edit has the same chance of ending up in the final sample.
    Memory use stays proportional to ``n``, never to the size of the file.

    ``idx`` is stable for a given country, ``n`` and ``seed``, which is what
    makes checkpoints work: a result stored under ``idx=57`` always refers to
    the same edit, as long as the sample is drawn the same way.
    """
    # data_file_path() resolves the file and rejects an unknown country name.
    path = data_file_path(country)

    if n is None:
        # Read everything.  Simple, but pandas builds the whole table in RAM.
        df = pd.read_csv(path, dtype=str)
    elif random_sample:
        # Draw a few extra edits: the tidying step below drops rows whose diff
        # is empty, and we still want to return exactly n of them.
        df = _reservoir_sample(path, n=int(n * 1.05) + 5, seed=seed)
    else:
        # nrows stops pandas after n data rows, so this is very fast.
        df = pd.read_csv(path, dtype=str, nrows=n)

    df = _tidy(df, country=country)

    if n is not None and len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    return df


def _reservoir_sample(path: Path, n: int, seed: int) -> pd.DataFrame:
    """Read a CSV and keep a uniform random sample of ``n`` data rows.

    How it works: the first ``n`` rows fill the reservoir.  Every later row
    takes the place of a random reservoir row with probability ``n / i``.  At
    the end, each row of the file has had the same probability of being kept,
    and only ``n`` rows are ever held in memory.
    """
    rng = random.Random(seed)
    reservoir: list[list[str]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        # csv.reader handles quoted fields, which matters here: a diff contains
        # line breaks, so one CSV row can span many physical lines in the file.
        reader = csv.reader(handle)
        header = next(reader)

        for i, row in enumerate(reader):
            if len(reservoir) < n:
                reservoir.append(row)
                continue
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = row

    return pd.DataFrame(reservoir, columns=header)


def _tidy(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """Clean the raw table and make sure the expected columns are there."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"The file for {country} is missing the columns {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Drop edits without diff text: they cannot be classified.
    df = df[df["changed_version"].notna() & (df["changed_version"].str.strip() != "")].copy()

    # Strip whitespace around the short text columns, so "  Kyiv  " and "Kyiv"
    # behave the same way everywhere downstream.
    for column in ("page_title", "section", "comment", "user"):
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()

    # Adding a plain integer index makes results easy to join back to the
    # source rows, and it is the key used by the checkpoint files.
    df = df.reset_index(drop=True)
    df.insert(0, "idx", df.index)
    return df


def sample_revisions(df: pd.DataFrame, n: int = 10, seed: int = 42) -> pd.DataFrame:
    """Return ``n`` random rows of ``df``, reproducing the same draw each time.

    ``random_state=seed`` makes the draw reproducible: run the same cell twice
    and you look at the same edits, which matters when comparing two models.
    """
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------
# Turning one row into LLM input
# --------------------------------------------------------------------------


def text(value) -> str:
    """Return a clean string, treating missing values as an empty string.

    Why this helper exists: an empty cell in a CSV becomes ``NaN`` in pandas,
    which is a *float*, not a string.  Writing ``if comment:`` on a NaN cell
    therefore succeeds and the prompt ends up containing the literal text
    "nan".  This function prevents that whole class of bug.

    It works for values coming from a DataFrame cell or from a plain dict.
    """
    if value is None:
        return ""
    # pandas stores missing values as float NaN, which is the only float that
    # is not equal to itself.
    if isinstance(value, float) and value != value:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def metadata_lines(edit) -> list[str]:
    """Return the metadata header lines for one edit.

    The order is fixed and matches the main CROSS2026 pipeline:

        Comment: <edit summary>
        Section: <section of the article>
        Page title: <article title>

    Empty fields are skipped, so the model never reads an empty label.
    """
    lines = []
    comment = text(_get(edit, "comment"))
    section = text(_get(edit, "section"))
    page_title = text(_get(edit, "page_title"))

    if comment:
        lines.append(f"Comment: {comment}")
    if section:
        lines.append(f"Section: {section}")
    if page_title:
        lines.append(f"Page title: {page_title}")
    return lines


def build_llm_input(edit, max_chars: int | None = None) -> str:
    """Build the complete text block we send to the LLM for one edit.

    The result looks like::

        Classify this Wikipedia edit:

        Comment: fixed typo
        Section: History
        Page title: Kyiv
        --- old revision
        +++ new revision
        @@ -3,1 +3,1 @@
        -Kiev was founded
        +Kyiv was founded

    Parameters
    ----------
    edit:
        One row of the DataFrame, or any object with the right keys (a dict
        works too).
    max_chars:
        Optional cap on the diff length.  When a diff is longer, it is cut and
        a clear marker is appended, so the model knows the text is incomplete.
    """
    parts = ["Classify this Wikipedia edit:\n"]
    parts.extend(metadata_lines(edit))

    diff = text(_get(edit, "changed_version"))
    parts.append(truncate(diff, max_chars) if max_chars else diff)
    return "\n".join(parts)


def truncate(diff: str, max_chars: int) -> str:
    """Cut ``diff`` after ``max_chars`` characters and mark it as truncated.

    Very long diffs can exceed the context window of a model, and the API then
    returns an error.  Cutting the text ourselves gives a readable answer and
    lets the run continue.  The marker is added so the model, and you, know
    that the edit continues beyond what is shown.
    """
    if len(diff) <= max_chars:
        return diff
    return diff[:max_chars] + "\n\n[... diff truncated ...]"


def _get(edit, key: str):
    """Read one field from a DataFrame row or from a plain dict.

    ``edit[key]`` works for both, but a missing column raises a KeyError with a
    message that is hard to read.  Going through this helper keeps the error
    clear: a missing field is simply treated as empty.
    """
    try:
        return edit[key]
    except (KeyError, IndexError, TypeError):
        return ""
