# PreCog task, The Ghost in the Machine

The different tasks have been split into different Jupyter notebooks for each task, except for task 2, in which the finetuning notebook has been further split into a different one. 

books/ folder has the chosen dataset from the Gutenberg project, and other texts like the LLM generated training set as well the output of the genetic algorithm.

All installed python packages have been put into requirements.txt. Python 3.13 was used for all the codes, **NOT** Python 3.14

## Setup for running the code:
1. [GloVe vector embeddings](https://nlp.stanford.edu/data/glove.6B.zip) - extract in folder named 'glove'

2. Install python packages
``` bash
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Setup .env file, with your gemini API key, env variable must be named GEMINI_API_KEY