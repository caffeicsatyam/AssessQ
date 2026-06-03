"""
agent.py — AssessQ backend
RAG pipeline (BM25 + optional dense) + LLM-powered conversational layer.

Conversation storage strategy
──────────────────────────────
• StreamlitChatMessageHistory  — stores messages directly in st.session_state so
  they survive Streamlit reruns automatically (no separate store dict needed).
• RunnableWithMessageHistory   — wraps the LLM chain and injects the history into
  every prompt automatically, keyed by session_id.
• For production / multi-user  — swap StreamlitChatMessageHistory for
  RedisChatMessageHistory or SQLChatMessageHistory (one-line change, see bottom).
"""

import argparse
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
import numpy as np
from dotenv import load_dotenv

# Suppress noisy deprecation warnings from upstream libraries
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message="Accessing `__path__`", module="transformers")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

load_dotenv()

# ── env / defaults ────────────────────────────────────────────────────────────
DEFAULT_CATALOG_PATH = Path(os.getenv("CATALOG_PATH", "data/catalog.json"))
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", 20))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", 20))
RRF_K = int(os.getenv("RRF_K", 60))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 5))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# ── data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Assessment:
    name: str
    url: str
    description: str
    job_levels: List[str]
    languages: List[str]
    assessment_length: Optional[str]
    test_types: List[str]
    remote_testing: bool


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> List[Assessment]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return [
        Assessment(
            name=row.get("name", "").strip(),
            url=row.get("url", "").strip(),
            description=row.get("description", "").strip(),
            job_levels=list(row.get("job_levels") or []),
            languages=list(row.get("languages") or []),
            assessment_length=(row.get("assessment_length") or "").strip() or None,
            test_types=list(row.get("test_types") or []),
            remote_testing=bool(row.get("remote_testing")),
        )
        for row in rows
        if row.get("name")
    ]


# ── text helpers ──────────────────────────────────────────────────────────────

def assessment_to_text(a: Assessment) -> str:
    return f"""
Assessment Name:
{a.name}

Description:
{a.description}

Job Levels:
{", ".join(a.job_levels) or "Not specified"}

Languages:
{", ".join(a.languages) or "Not specified"}

Assessment Length:
{a.assessment_length or "Not specified"}

Test Types:
{", ".join(a.test_types) or "Not specified"}

Remote Testing:
{a.remote_testing}
""".strip()


def assessment_to_metadata(a: Assessment) -> dict:
    return {
        "assessment_name": a.name,
        "url": a.url,
        "remote_testing": a.remote_testing,
        "job_levels": a.job_levels,
        "languages": a.languages,
        "test_types": a.test_types,
        "assessment_length": a.assessment_length,
    }


def metadata_for_chroma(a: Assessment) -> dict:
    meta = assessment_to_metadata(a)
    return {
        key: json.dumps(value) if isinstance(value, list) else (value or "")
        for key, value in meta.items()
    }


def metadata_from_chroma(meta: dict) -> dict:
    decoded = dict(meta)
    for key in ("job_levels", "languages", "test_types"):
        value = decoded.get(key)
        if isinstance(value, str):
            try:
                decoded[key] = json.loads(value)
            except json.JSONDecodeError:
                decoded[key] = [value] if value else []
    decoded["assessment_length"] = decoded.get("assessment_length") or None
    return decoded


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9+#.]+", text.lower())


# ── vector store ──────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR, collection_name: str = "assessment_store"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logging.info("Initializing embedding model.")
            self._embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._embedder

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        prefixed = [f"Represent this sentence for searching relevant passages: {text}" for text in texts]
        return self._get_embedder().encode(
            prefixed, convert_to_numpy=True, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )

    def embed_query(self, query: str) -> List[float]:
        return self._get_embedder().encode(
            f"Represent this question for searching relevant passages: {query}", normalize_embeddings=True
        ).tolist()

    def get_collection(self):
        return self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

    def index_assessments(self, assessments: List[Assessment]):
        collection = self.get_collection()
        if not assessments:
            return collection
        documents = [assessment_to_text(a) for a in assessments]
        ids = [f"assessment_{i}" for i, _ in enumerate(assessments)]
        collection.upsert(
            ids=ids, documents=documents,
            embeddings=self.embed(documents).tolist(),
            metadatas=[metadata_for_chroma(a) for a in assessments],
        )
        return collection

    def dense_search(self, query: str, top_k: int = DENSE_TOP_K) -> List[dict]:
        collection = self.get_collection()
        if collection.count() == 0:
            return []
        results = collection.query(
            query_embeddings=[self.embed_query(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            {"document": doc, "metadata": metadata_from_chroma(meta), "distance": dist, "source": "dense"}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]


# ── BM25 ──────────────────────────────────────────────────────────────────────

class SimpleBM25Okapi:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = np.array([len(doc) for doc in corpus], dtype=float)
        self.avgdl = float(np.mean(self.doc_lengths)) if len(self.doc_lengths) else 0.0
        self.term_frequencies: List[dict] = []
        document_frequencies: dict = {}
        for document in corpus:
            frequencies: dict = {}
            for term in document:
                frequencies[term] = frequencies.get(term, 0) + 1
            self.term_frequencies.append(frequencies)
            for term in frequencies:
                document_frequencies[term] = document_frequencies.get(term, 0) + 1
        corpus_size = len(corpus)
        self.idf = {
            term: np.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequencies.items()
        }

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=float)
        if not query_tokens or not self.corpus:
            return scores
        for index, frequencies in enumerate(self.term_frequencies):
            doc_length = self.doc_lengths[index]
            for term in query_tokens:
                tf = frequencies.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * doc_length / (self.avgdl or 1.0))
                scores[index] += self.idf.get(term, 0.0) * (tf * (self.k1 + 1) / denom)
        return scores


class BM25Index:
    def __init__(self):
        try:
            from rank_bm25 import BM25Okapi
        except ModuleNotFoundError:
            BM25Okapi = SimpleBM25Okapi
        self._bm25_cls = BM25Okapi
        self.bm25 = None
        self.documents: List[str] = []
        self.metadatas: List[dict] = []

    def index(self, assessments: List[Assessment]):
        self.documents = [assessment_to_text(a) for a in assessments]
        self.metadatas = [assessment_to_metadata(a) for a in assessments]
        self.bm25 = self._bm25_cls([tokenize(doc) for doc in self.documents])

    def search(self, query: str, top_k: int = BM25_TOP_K) -> List[dict]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"document": self.documents[i], "metadata": self.metadatas[i], "score": float(scores[i]), "source": "bm25"}
            for i in top_indices if scores[i] > 0
        ]


# ── RRF fusion ────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: List[dict],
    bm25_results: List[dict],
    k: int = RRF_K,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[dict]:
    scores: dict = {}
    docs: dict = {}

    def add_results(results: Iterable[dict], weight: float):
        for rank, result in enumerate(results):
            doc_id = result["metadata"]["assessment_name"]
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
            docs.setdefault(doc_id, result)

    add_results(dense_results, dense_weight)
    add_results(bm25_results, sparse_weight)
    return [
        {**docs[doc_id], "rrf_score": score}
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


# ── reranker ──────────────────────────────────────────────────────────────────

class ReRanker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.model = None
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def get_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        return self.model

    def rerank(self, query: str, candidates: List[dict], top_n: int = RERANK_TOP_N) -> List[dict]:
        if not candidates:
            return []
        if not self.enabled:
            return candidates[:top_n]
        try:
            scores = self.get_model().predict([(query, c["document"]) for c in candidates])
        except ModuleNotFoundError:
            return candidates[:top_n]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_n]]


# ── hard filter ───────────────────────────────────────────────────────────────

class HardFilter:
    def __init__(self, job_levels=None, test_types=None, languages=None, remote_testing=None):
        self.job_levels = {v.lower() for v in (job_levels or [])}
        self.test_types = {v.upper() for v in (test_types or [])}
        self.languages = {v.lower() for v in (languages or [])}
        self.remote_testing = remote_testing

    def filter(self, candidates: List[dict]) -> List[dict]:
        filtered = []
        for c in candidates:
            meta = c.get("metadata", {})
            jl = {v.lower() for v in meta.get("job_levels", [])}
            tt = {v.upper() for v in meta.get("test_types", [])}
            la = {v.lower() for v in meta.get("languages", [])}
            if self.job_levels and not self.job_levels.intersection(jl):
                continue
            if self.test_types and not self.test_types.intersection(tt):
                continue
            if self.languages and not self.languages.intersection(la):
                continue
            if self.remote_testing is not None and meta.get("remote_testing") != self.remote_testing:
                continue
            filtered.append(c)
        return filtered


# ── core recommender ──────────────────────────────────────────────────────────

class AssessmentRecommender:
    def __init__(self, assessments: List[Assessment], use_dense: bool = True, use_reranker: bool = True, persist_dir: str = DEFAULT_PERSIST_DIR):
        self.assessments = assessments
        self.bm25 = BM25Index()
        self.bm25.index(assessments)
        self.vector_store = None
        if use_dense:
            try:
                self.vector_store = VectorStore(persist_dir=persist_dir)
            except ModuleNotFoundError as exc:
                logging.warning("Dense search disabled: %s", exc.name)
        self.reranker = ReRanker(enabled=use_reranker)

    def ensure_dense_index(self):
        if self.vector_store is None:
            return
        collection = self.vector_store.get_collection()
        if collection.count() == 0:
            try:
                self.vector_store.index_assessments(self.assessments)
            except ModuleNotFoundError as exc:
                logging.warning("Dense search disabled: %s", exc.name)
                self.vector_store = None

    def retrieve(self, query: str, filter_params: Optional[dict] = None, top_k: int = BM25_TOP_K) -> List[dict]:
        dense_results: List[dict] = []
        if self.vector_store is not None:
            self.ensure_dense_index()
            if self.vector_store is not None:
                dense_results = self.vector_store.dense_search(query, top_k=top_k)
        bm25_results = self.bm25.search(query, top_k=top_k)
        candidates = reciprocal_rank_fusion(dense_results, bm25_results)
        if filter_params:
            candidates = HardFilter(**filter_params).filter(candidates)
        return candidates

    def query(self, query: str, filter_params: Optional[dict] = None, top_k: int = BM25_TOP_K, top_n: int = RERANK_TOP_N) -> List[dict]:
        candidates = self.retrieve(query, filter_params=filter_params, top_k=top_k)
        return self.reranker.rerank(query, candidates, top_n=top_n)


# ── LLM conversation layer ────────────────────────────────────────────────────
#
# Uses RunnableWithMessageHistory + StreamlitChatMessageHistory so that:
#   1. Full conversation history is injected into every LLM call automatically.
#   2. Messages are stored in st.session_state — survive Streamlit reruns.
#   3. Switching to Redis/SQL for production is a one-line change (see bottom).

class ConversationalRecommender:
    """
    Wraps AssessmentRecommender with an LLM that:
      • Asks clarifying questions before recommending.
      • Understands refinement instructions (drop/add/replace).
      • Has full conversation memory via StreamlitChatMessageHistory.
    """

    SYSTEM_PROMPT = """You are AssessQ, an expert SHL assessment advisor.

    Your job:
    1. Ask 1-2 targeted clarifying questions if the role description is vague (backend vs frontend,
    seniority level, IC vs tech-lead). Do NOT ask more than 2 questions before recommending.
    2. Once you have enough context, output a JSON block with this exact shape and nothing else after it:

    ```json
    {{
    "action": "recommend" | "clarify" | "refine" | "confirm",
    "reply": "<your conversational reply to show the user>",
    "query": "<enriched search query combining all context so far, or null>",
    "drops": ["<term to remove>"],
    "adds": ["<term to add>"]
    }}
    ```

    Rules:
    - action=clarify  → you need more info; reply contains your question; query/drops/adds are null/[].
    - action=recommend → you have enough context; query is the full enriched retrieval string.
    - action=refine   → user asked to drop/add/replace something; populate drops and/or adds.
    - action=confirm  → user confirmed the shortlist as final.
    - Always build the query by combining ALL relevant details from the entire conversation, not just the latest message.
    - Never invent assessment names. Only retrieve from the catalog.
    - Keep replies concise and professional.
    """
    
    def __init__(self, recommender: AssessmentRecommender):
        self.recommender = recommender
        self._chain = None
        self._history_store: dict[str, InMemoryChatMessageHistory] = {}

    def _get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._history_store:
            self._history_store[session_id] = InMemoryChatMessageHistory()
        return self._history_store[session_id]

    def _get_llm(self):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.0,
            groq_api_key=GROQ_API_KEY
        )


    # Max Hitory Stored 
    MAX_HISTORY_TURNS = 5

    def chat(self, user_input: str, session_id: str, filter_params: Optional[dict] = None, top_k: int = BM25_TOP_K, top_n: int = RERANK_TOP_N) -> dict:
        history = self._get_history(session_id)
        recent = history.messages[-(self.MAX_HISTORY_TURNS * 2):]

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            *recent,
            HumanMessage(content=user_input),
        ]

        llm = self._get_llm()
        raw = llm.invoke(messages)

        # Save turn to full history (not truncated)
        history.add_user_message(user_input)
        history.add_ai_message(raw.content)

        parsed = self._parse_llm_output(raw.content)
        
        action = parsed.get("action")
        query = parsed.get("query")
        
        if action in ("recommend", "refine") and query:
            results = self.recommender.query(
                query=query,
                filter_params=filter_params,
                top_k=top_k,
                top_n=top_n
            )

            if results:
                results = self._validate_results(llm, query, results, recent)

            parsed["results"] = results if results else []
        else:
            parsed["results"] = None

        return parsed

    VALIDATION_PROMPT = """You are a relevance judge for SHL assessment recommendations.

Conversation so far (most recent messages):
{history}

Enriched search query: {query}

Retrieved assessments:
{table}

Task: Review each numbered assessment above against the conversation context and query.
Return ONLY a JSON array of the assessment numbers (1-indexed) that are genuinely
relevant to what the user needs. Drop any that are clearly unrelated.

Example output: [1, 3, 5]
If none are relevant: []

Return ONLY the JSON array, nothing else."""

    def _validate_results(
        self,
        llm,
        query: str,
        results: List[dict],
        recent_messages: list,
    ) -> List[dict]:
        """Use the LLM to prune irrelevant results from the retrieval set."""
        # Build a readable conversation snippet for context
        history_lines = []
        for msg in recent_messages[-6:]:          # last 3 turns max
            role = getattr(msg, "type", "unknown")
            # Truncate long AI responses (they contain JSON blocks)
            content = msg.content[:300] if hasattr(msg, "content") else str(msg)[:300]
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines) if history_lines else "(first turn)"

        prompt = self.VALIDATION_PROMPT.format(
            history=history_text,
            query=query,
            table=format_table(results),
        )

        try:
            validation = llm.invoke(prompt)
            match = re.search(r"\[[\d\s,]*\]", validation.content)
            if match:
                valid_indices = set(json.loads(match.group(0)))
                filtered = [r for i, r in enumerate(results, 1) if i in valid_indices]
                return filtered if filtered else results   # never return empty if we had results
        except Exception:
            pass

        return results   # fallback: keep all if validation fails

    @staticmethod
    def _parse_llm_output(content: str) -> dict:
        """Extract and parse the JSON block from the LLM response."""
        match = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
        if not match:
            match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if "```" in content else match.group(0))
            except json.JSONDecodeError:
                pass
        return {
            "action": "clarify", 
            "reply": content, 
            "query": None, 
            "drops": [], 
            "adds": []
        }


# ── formatting helpers ────────────────────────────────────────────────────────

def duration_minutes(value: Optional[str]) -> str:
    if not value:
        return "-"
    match = re.search(r"(\d+)", value)
    return f"{match.group(1)} minutes" if match else value.replace("Approximate Completion Time in minutes =", "").strip()


def compact_languages(languages: List[str], limit: int = 4) -> str:
    if not languages:
        return "-"
    if len(languages) <= limit:
        return ", ".join(languages)
    return f"{', '.join(languages[:limit])} (+{len(languages) - limit} more)"


def keys_for_types(test_types: List[str]) -> str:
    labels = [TEST_TYPE_LABELS.get(v, v) for v in test_types]
    return ", ".join(labels) if labels else "-"


def format_table(results: List[dict]) -> str:
    lines = [
        "| # | Name | Test Type | Keys | Duration | Languages | URL |",
        "|---|------|-----------|------|----------|-----------|-----|",
    ]
    for index, result in enumerate(results, 1):
        meta = result["metadata"]
        test_types = meta.get("test_types", [])
        lines.append(
            "| {i} | {name} | {types} | {keys} | {dur} | {lang} | {url} |".format(
                i=index,
                name=meta.get("assessment_name", "-"),
                types=", ".join(test_types) or "-",
                keys=keys_for_types(test_types),
                dur=duration_minutes(meta.get("assessment_length")),
                lang=compact_languages(meta.get("languages", [])),
                url=meta.get("url", "-"),
            )
        )
    return "\n".join(lines)


def format_context_item(index: int, result: dict) -> str:
    meta = result["metadata"]
    return f"""
Assessment #{index}
Name: {meta.get("assessment_name", "Unknown")}
Job Levels: {", ".join(meta.get("job_levels", [])) or "Not specified"}
Languages: {", ".join(meta.get("languages", [])) or "Not specified"}
Test Types: {", ".join(meta.get("test_types", [])) or "Not specified"}
Remote Testing: {meta.get("remote_testing")}
URL: {meta.get("url")}
Description: {result["document"]}
""".strip()


# ── CLI ───────────────────────────────────────────────────────────────────────

def split_csv(values: Optional[str]) -> Optional[List[str]]:
    if not values:
        return None
    return [v.strip() for v in values.split(",") if v.strip()]


def parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    if value.lower() in {"true", "yes", "1"}:
        return True
    if value.lower() in {"false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Catalog-based SHL assessment recommender.")
    parser.add_argument("query")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--top-k", type=int, default=BM25_TOP_K)
    parser.add_argument("--top-n", type=int, default=RERANK_TOP_N)
    parser.add_argument("--job-levels")
    parser.add_argument("--test-types")
    parser.add_argument("--languages")
    parser.add_argument("--remote-testing", type=parse_bool)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--no-dense", action="store_false", dest="dense")
    parser.add_argument("--no-rerank", action="store_false", dest="rerank")
    parser.add_argument("--table", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    assessments = load_catalog(Path(args.catalog))
    recommender = AssessmentRecommender(assessments, use_dense=args.dense, use_reranker=args.rerank)
    filter_params = {k: v for k, v in {
        "job_levels": split_csv(args.job_levels),
        "test_types": split_csv(args.test_types),
        "languages": split_csv(args.languages),
        "remote_testing": args.remote_testing,
    }.items() if v is not None}
    results = recommender.query(args.query, filter_params=filter_params or None, top_k=args.top_k, top_n=args.top_n)
    if args.table:
        print(format_table(results))
    else:
        print(json.dumps([r["metadata"] for r in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())