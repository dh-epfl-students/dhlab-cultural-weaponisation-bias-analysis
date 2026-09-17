# SEMESTER PROJECT: Bias Analysis of cultural heritage manipulation classification of wikipedia edits by LLMs

## Installation

* **Install packages** with requirements.txt
* **Get an API key** for accessing the RCP endpoint and add it as an environment variable

## Data

The revision CSVs are about 2.7 GB per country and are not in this repository.
`preprocessing.py` looks for them in `data/Ukraine_raw.csv` or `data/Ukraine/Ukraine_raw.csv`

Reading a whole country file takes about 20 seconds and a lot of memory, so use
`load_revisions(country, n=..., seed=...)` to work on a reproducible random sample.

## Initial commit

The scripts contain content derived [from the main repo](https://github.com/dhlab-epfl/cultural-heritage-weaponisation-extraction) which allows you to load data from the raw csvs (not present in this repo ofc) and perform classification. 
You will also find simple notebooks for data exploration and some quick classification trials. 

FILES:
- **checkpoint.py** handles checkpointing for classification if needed
- **classification.py** handles classification with checkpointing, backwards compatibility with older code, and some quick regex stuff to make sure the results are parsed properly. 
- **preprocessing.py** loads the raw CSV files, normalises the metadata (`page_title`, `section`, `comment`) and builds the exact text block sent to the model.
- **utils.py** misc stuff for accessing the RCP endpoint, loading API key.

NOTEBOOKS (read them in this order):
- **0-1-exploration.ipynb** loads the data and looks at the edits and their metadata. Calls no model, so it runs without an API key.
- **0-2-classification.ipynb** classifies 10 random Ukrainian edits and saves the results.
