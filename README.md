# NCAP RAG

Small local RAG helper for querying Euro NCAP PDF documents.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put source PDFs in `data/raw/`, then build the chunks and FAISS index:

```bash
python -m ncap_rag.ingest
```

## Ask a question

Set AWS Bedrock credentials and `BEDROCK_MODEL_ID`, then run:

```bash
python -m ncap_rag.ask "In what category is alcohol intoxication?"
```

To inspect retrieved passages without calling Bedrock:

```bash
python -m ncap_rag.retrieve "In what category is alcohol intoxication?"
```
