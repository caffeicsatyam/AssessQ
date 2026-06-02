import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CATALOG_PATH = Path(os.getenv("CATALOG_PATH", "data/catalog.json"))
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", 20))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", 20))
RRF_K = int(os.getenv("RRF_K", 60))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 5))


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


class VectorStore:
    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = "assessment_store",
    ):
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

        prefixed = [
            f"Represent this sentence for searching relevant passages: {text}"
            for text in texts
        ]
        return self._get_embedder().encode(
            prefixed,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> List[float]:
        return self._get_embedder().encode(
            f"Represent this question for searching relevant passages: {query}",
            normalize_embeddings=True,
        ).tolist()

    def get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_assessments(self, assessments: List[Assessment]):
        collection = self.get_collection()
        if not assessments:
            logging.warning("No assessments to index.")
            return collection

        documents = [assessment_to_text(a) for a in assessments]
        ids = [f"assessment_{i}" for i, _ in enumerate(assessments)]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=self.embed(documents).tolist(),
            metadatas=[metadata_for_chroma(a) for a in assessments],
        )
        logging.info("Indexed %s assessments in Chroma.", len(documents))
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
            {
                "document": doc,
                "metadata": metadata_from_chroma(meta),
                "distance": distance,
                "source": "dense",
            }
            for doc, meta, distance in zip(documents, metadatas, distances)
        ]


class SimpleBM25Okapi:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = np.array([len(doc) for doc in corpus], dtype=float)
        self.avgdl = float(np.mean(self.doc_lengths)) if len(self.doc_lengths) else 0.0
        self.term_frequencies: List[dict[str, int]] = []
        document_frequencies: dict[str, int] = {}

        for document in corpus:
            frequencies: dict[str, int] = {}
            for term in document:
                frequencies[term] = frequencies.get(term, 0) + 1
            self.term_frequencies.append(frequencies)
            for term in frequencies:
                document_frequencies[term] = document_frequencies.get(term, 0) + 1

        corpus_size = len(corpus)
        self.idf = {
            term: np.log(1 + (corpus_size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=float)
        if not query_tokens or not self.corpus:
            return scores

        for index, frequencies in enumerate(self.term_frequencies):
            doc_length = self.doc_lengths[index]
            for term in query_tokens:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / (self.avgdl or 1.0)
                )
                scores[index] += self.idf.get(term, 0.0) * (
                    term_frequency * (self.k1 + 1) / denominator
                )
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
            logging.warning("BM25 index is not initialized.")
            return []

        scores = self.bm25.get_scores(tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "document": self.documents[i],
                "metadata": self.metadatas[i],
                "score": float(scores[i]),
                "source": "bm25",
            }
            for i in top_indices
            if scores[i] > 0
        ]


def reciprocal_rank_fusion(
    dense_results: List[dict],
    bm25_results: List[dict],
    k: int = RRF_K,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[dict]:
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    def add_results(results: Iterable[dict], weight: float):
        for rank, result in enumerate(results):
            doc_id = result["metadata"]["assessment_name"]
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank + 1)
            docs.setdefault(doc_id, result)

    add_results(dense_results, dense_weight)
    add_results(bm25_results, sparse_weight)

    return [
        {**docs[doc_id], "rrf_score": score}
        for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


class ReRanker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.model = None
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def get_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder

            logging.info("Initializing re-ranker model.")
            self.model = CrossEncoder(self.model_name)
        return self.model

    def rerank(self, query: str, candidates: List[dict], top_n: int = RERANK_TOP_N) -> List[dict]:
        if not candidates:
            return []
        if not self.enabled:
            return candidates[:top_n]

        try:
            scores = self.get_model().predict([(query, c["document"]) for c in candidates])
        except ModuleNotFoundError as exc:
            logging.warning("Reranker disabled because %s is not installed.", exc.name)
            return candidates[:top_n]
        ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
        return [{**candidate, "rerank_score": float(score)} for candidate, score in ranked[:top_n]]


class HardFilter:
    def __init__(
        self,
        job_levels: Optional[List[str]] = None,
        test_types: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        remote_testing: Optional[bool] = None,
    ):
        self.job_levels = {value.lower() for value in job_levels or []}
        self.test_types = {value.upper() for value in test_types or []}
        self.languages = {value.lower() for value in languages or []}
        self.remote_testing = remote_testing

    def filter(self, candidates: List[dict]) -> List[dict]:
        filtered = []
        for candidate in candidates:
            meta = candidate.get("metadata", {})
            job_levels = {value.lower() for value in meta.get("job_levels", [])}
            test_types = {value.upper() for value in meta.get("test_types", [])}
            languages = {value.lower() for value in meta.get("languages", [])}

            if self.job_levels and not self.job_levels.intersection(job_levels):
                continue
            if self.test_types and not self.test_types.intersection(test_types):
                continue
            if self.languages and not self.languages.intersection(languages):
                continue
            if self.remote_testing is not None and meta.get("remote_testing") != self.remote_testing:
                continue
            filtered.append(candidate)

        return filtered


class AssessmentRecommender:
    def __init__(
        self,
        assessments: List[Assessment],
        use_dense: bool = True,
        use_reranker: bool = True,
        persist_dir: str = DEFAULT_PERSIST_DIR,
    ):
        self.assessments = assessments
        self.bm25 = BM25Index()
        self.bm25.index(assessments)
        self.vector_store = None
        if use_dense:
            try:
                self.vector_store = VectorStore(persist_dir=persist_dir)
            except ModuleNotFoundError as exc:
                logging.warning("Dense search disabled because %s is not installed.", exc.name)
        self.reranker = ReRanker(enabled=use_reranker)

    def ensure_dense_index(self):
        if self.vector_store is None:
            return
        collection = self.vector_store.get_collection()
        if collection.count() == 0:
            try:
                self.vector_store.index_assessments(self.assessments)
            except ModuleNotFoundError as exc:
                logging.warning("Dense search disabled because %s is not installed.", exc.name)
                self.vector_store = None

    def retrieve(
        self,
        query: str,
        filter_params: Optional[dict] = None,
        top_k: int = BM25_TOP_K,
    ) -> List[dict]:
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

    def query(
        self,
        query: str,
        filter_params: Optional[dict] = None,
        top_k: int = BM25_TOP_K,
        top_n: int = RERANK_TOP_N,
    ) -> List[dict]:
        candidates = self.retrieve(query, filter_params=filter_params, top_k=top_k)
        return self.reranker.rerank(query, candidates, top_n=top_n)

    def generate_response(
        self,
        query: str,
        llm_fn: Callable[[str], str],
        filter_params: Optional[dict] = None,
        top_k: int = BM25_TOP_K,
        top_n: int = RERANK_TOP_N,
    ) -> str:
        results = self.query(query, filter_params=filter_params, top_k=top_k, top_n=top_n)
        if not results:
            return "No relevant assessments found."

        context = "\n\n".join(format_context_item(i, item) for i, item in enumerate(results, 1))
        prompt = f"""
You are an SHL assessment recommendation expert.

User Request:
{query}

Candidate Assessments:
{context}

Instructions:
- Recommend the most suitable assessments.
- Use only the assessments provided.
- Do not invent assessments.
- Explain why each recommendation fits.
- Mention relevant skills measured.
- Mention important constraints like language or duration.
- Rank recommendations from best to worst.

Return a professional recommendation report.
""".strip()
        return llm_fn(prompt)


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
    labels = [TEST_TYPE_LABELS.get(value, value) for value in test_types]
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
            "| {index} | {name} | {types} | {keys} | {duration} | {languages} | {url} |".format(
                index=index,
                name=meta.get("assessment_name", "-"),
                types=", ".join(test_types) or "-",
                keys=keys_for_types(test_types),
                duration=duration_minutes(meta.get("assessment_length")),
                languages=compact_languages(meta.get("languages", [])),
                url=meta.get("url", "-"),
            )
        )
    return "\n".join(lines)


def split_csv(values: Optional[str]) -> Optional[List[str]]:
    if not values:
        return None
    return [value.strip() for value in values.split(",") if value.strip()]


def parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Catalog-based SHL assessment recommender.")
    parser.add_argument("query", help="Role, job description, or assessment need to search for.")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG_PATH), help="Path to catalog JSON.")
    parser.add_argument("--top-k", type=int, default=BM25_TOP_K, help="Candidates to retrieve before reranking.")
    parser.add_argument("--top-n", type=int, default=RERANK_TOP_N, help="Recommendations to return.")
    parser.add_argument("--job-levels", help="Comma-separated job level filter.")
    parser.add_argument("--test-types", help="Comma-separated test type filter, e.g. K,S,P.")
    parser.add_argument("--languages", help="Comma-separated language filter.")
    parser.add_argument("--remote-testing", type=parse_bool, help="Filter remote testing true/false.")
    parser.add_argument("--dense", action="store_true", help="Enable Chroma dense retrieval and embedding model load.")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking.")
    parser.add_argument("--no-dense", action="store_false", dest="dense", help=argparse.SUPPRESS)
    parser.add_argument("--no-rerank", action="store_false", dest="rerank", help=argparse.SUPPRESS)
    parser.add_argument("--table", action="store_true", help="Print recommendations as a markdown table.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    assessments = load_catalog(Path(args.catalog))
    recommender = AssessmentRecommender(
        assessments,
        use_dense=args.dense,
        use_reranker=args.rerank,
    )
    filter_params = {
        "job_levels": split_csv(args.job_levels),
        "test_types": split_csv(args.test_types),
        "languages": split_csv(args.languages),
        "remote_testing": args.remote_testing,
    }
    filter_params = {key: value for key, value in filter_params.items() if value is not None}

    results = recommender.query(
        args.query,
        filter_params=filter_params or None,
        top_k=args.top_k,
        top_n=args.top_n,
    )
    if args.table:
        print(format_table(results))
    else:
        print(json.dumps([result["metadata"] for result in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
