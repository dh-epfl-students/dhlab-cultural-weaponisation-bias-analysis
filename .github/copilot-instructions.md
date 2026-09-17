# Copilot instructions: bias analysis student project

Standalone MSc semester project, part of CROSS2026 but **not** part of the main
`data-extraction-rework/` pipeline. Never import from that folder.

Goal: study the bias of LLMs that classify Wikipedia edits about Armenian and Ukrainian
cultural heritage as potentially manipulative. Labels: `Potentially Manipulative`,
`Not Problematic`, `Ambiguous`.

## Layout

| File | Purpose |
|---|---|
| `preprocessing.py` | loads the revision CSVs, normalises metadata, builds the LLM input |
| `classification.py` | system prompt, async classification engine, parsing, export |
| `utils.py` | RCP endpoint, model name, API key, async client |
| `checkpoint.py` | JSONL append-only checkpointing for long runs |
| `0-1-exploration.ipynb`, `0-2-classification.ipynb` | starter notebooks, read in order |

Runtime folders, git-ignored: `data/`, `checkpoints/`, `outputs/`.

## Conventions

- Python 3.12. `encoding="utf-8"` everywhere, `utf-8-sig` for exported CSVs.
- This is **teaching code**: comment it, keep functions small, prefer clarity over cleverness.
- Avoid em dashes and en dashes in files in this folder.
- Every random draw takes a `seed`.
- One LLM request per edit, in parallel through a semaphore, with a hard timeout.
- Failures never stop a run: the result is stored with `label="Error"`.
- Long runs write a checkpoint; resume by passing the same `checkpoint_path`.
- Input layout sent to the model is `Comment:` / `Section:` / `Page title:` then the raw diff,
  empty fields skipped. Metadata helpers are NaN-safe.

## Data

The CSVs (about 2.7 GB per country) are not in the repository and are never committed. The loader
looks for `data/{Country}_raw.csv` or `data/{Country}/{Country}_raw.csv`; copy the CSV there and it
is found automatically. See the README for where to get the files.

## Model

Default `Qwen/Qwen3.6-35B-A3B` through `https://inference.rcp.epfl.ch/v1`. Qwen is a reasoning
model: the final answer is in `content`, the thinking is in `reasoning_content`.
