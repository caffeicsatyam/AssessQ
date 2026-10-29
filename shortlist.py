"""Deterministic helpers for editing an assessment shortlist.

The chat model remains responsible for recommendation quality, but simple list
edits should never depend on whether it happened to classify a turn correctly.
"""

import re
from typing import Iterable, List, Optional, Tuple


_DROP_RE = re.compile(
    r"\b(?:remove|delete|drop|exclude)\s+(?:the\s+)?(?P<term>.+?)"
    r"(?=\s+(?:and|then)\s+(?:add|include|insert)\b|\s*[,;]|$)",
    re.IGNORECASE,
)
_ADD_RE = re.compile(
    r"\b(?:add|include|insert)\s+(?:the\s+)?(?P<term>.+?)"
    r"(?=\s+(?:and|then)\s+(?:remove|delete|drop|exclude)\b|\s*[,;]|$)",
    re.IGNORECASE,
)
_REPLACE_RE = re.compile(
    r"\breplace\s+(?:the\s+)?(?P<drop>.+?)\s+with\s+(?:the\s+)?(?P<add>.+?)\s*$",
    re.IGNORECASE,
)
_GENERIC_SUFFIX_RE = re.compile(r"\s+(?:assessments?|tests?)\s*$", re.IGNORECASE)


def _term(value: str) -> str:
    """Normalize conversational phrasing into a catalog-searchable term."""
    value = re.sub(r"^(?:the|an|a)\s+", "", value.strip(), flags=re.IGNORECASE)
    return _GENERIC_SUFFIX_RE.sub("", value).strip()


def parse_shortlist_edit(message: str) -> Tuple[List[str], List[str]]:
    """Return explicit (drops, adds) requested in a natural-language command."""
    replacement = _REPLACE_RE.search(message)
    if replacement:
        return [_term(replacement.group("drop"))], [_term(replacement.group("add"))]

    drops = [_term(match.group("term")) for match in _DROP_RE.finditer(message)]
    adds = [_term(match.group("term")) for match in _ADD_RE.finditer(message)]
    return [term for term in drops if term], [term for term in adds if term]


def contains_term(result: dict, term: str) -> bool:
    meta = result["metadata"]
    text = " ".join(
        [
            meta.get("assessment_name", ""),
            result.get("document", ""),
            " ".join(meta.get("test_types", [])),
            " ".join(meta.get("languages", [])),
        ]
    ).lower()
    return term.lower().strip() in text


def name_contains_term(result: dict, term: str) -> bool:
    """Use the catalog name—not a broad description—to identify an explicit add."""
    name = result["metadata"].get("assessment_name", "")
    return term.lower().strip() in name.lower()


def unique_results(results: Iterable[dict]) -> List[dict]:
    seen = set()
    unique = []
    for result in results:
        name = result["metadata"].get("assessment_name")
        if name and name not in seen:
            seen.add(name)
            unique.append(result)
    return unique


def apply_shortlist_edit(
    current: Iterable[dict],
    drops: List[str],
    adds: List[str],
    retrieved: Optional[List[dict]],
    limit: int,
) -> List[dict]:
    """Merge requested additions and apply removals without reintroducing drops."""
    current = list(current)
    retrieved = retrieved or []
    current_names = {result["metadata"].get("assessment_name") for result in current}
    missing_adds = [
        term for term in adds if not any(contains_term(result, term) for result in current)
    ]
    requested_additions = [
        result
        for result in retrieved
        if result["metadata"].get("assessment_name") not in current_names
        and any(name_contains_term(result, term) for term in missing_adds)
    ]
    # Retrieval is only evidence for the explicitly requested additions.  Merging
    # every search hit would silently mutate a shortlist on a simple removal.
    merged = unique_results([*requested_additions, *current])
    if drops:
        merged = [
            result
            for result in merged
            if not any(contains_term(result, term) for term in drops)
        ]
    return merged[:limit]
