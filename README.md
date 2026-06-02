# Assessment Recommender

Catalog-only recommendation system for SHL-style assessment batteries.

The single source of truth is:

```text
data/catalog.json
```

The recommender now uses retrieval and ranking only:

```text
catalog.json -> normalize -> BM25 retrieval
             -> optional dense Chroma retrieval
             -> reciprocal-rank fusion
             -> optional cross-encoder reranking
             -> ranked recommendations
```

There is no graph-RAG dependency in the current implementation.

## Usage

Run the Streamlit chat app:

```powershell
streamlit run app.py
```

The Streamlit app keeps the active shortlist in `st.session_state`, so follow-up prompts can refine the previous answer:

```text
drop REST and add AWS and Docker
```

Run the default lightweight path with no embedding or reranker model load:

```powershell
python agent.py "Senior backend engineer with Core Java, Spring, SQL, AWS, and Docker" --top-n 7 --table
```

Use filters:

```powershell
python agent.py "healthcare admin HIPAA medical terminology word" --languages "English (USA)" --table
```

Use the optional dense/rerank path when dependencies and models are available:

```powershell
python agent.py "Senior backend engineer with Core Java, Spring, SQL, AWS, and Docker" --top-n 7 --table --dense --rerank
```

If ChromaDB, sentence-transformers, or rank-bm25 are unavailable, the CLI falls back to the built-in lexical retrieval path where possible.

## Output

Use `--table` for a markdown table with:

```text
#, Name, Test Type, Keys, Duration, Languages, URL
```

Without `--table`, the CLI prints JSON metadata for the selected assessments.

## Useful Options

- `--top-k`: number of candidates retrieved before reranking.
- `--top-n`: number of recommendations returned.
- `--job-levels`: comma-separated job level filter.
- `--test-types`: comma-separated SHL test type filter, for example `K,S,P`.
- `--languages`: comma-separated language filter.
- `--remote-testing`: `true` or `false`.
- `--dense`: enable Chroma and embedding model retrieval.
- `--rerank`: enable cross-encoder reranking.
