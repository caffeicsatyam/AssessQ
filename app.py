import re
from collections import Counter
from typing import Iterable, List, Optional

import pandas as pd
import streamlit as st
from agent import AssessmentRecommender
from agent import  TEST_TYPE_LABELS, format_table, load_catalog


CONFIRMATION_RE = re.compile(
    r"\b(perfect|looks good|that covers it|lock|locking|final|done|yes|correct|great)\b",
    re.IGNORECASE,
)
REPLACE_RE = re.compile(
    r"\breplace\s+(.+?)\s+with\s+(.+)$",
    re.IGNORECASE,
)
DROP_RE = re.compile(
    r"\b(?:drop|remove|exclude|skip)\s+(.+?)(?=\b(?:and\s+)?(?:add|include|replace|with)\b|$)",
    re.IGNORECASE,
)
ADD_RE = re.compile(
    r"\b(?:add|include)\s+(.+?)(?=\b(?:and\s+)?(?:drop|remove|exclude|skip|replace)\b|$)",
    re.IGNORECASE,
)
SMALL_TALK_RE = re.compile(
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|thanks|thank\s+you|help|what\s+can\s+you\s+do)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


st.set_page_config(
    page_title="AssessQ",
    page_icon="AQ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 750;
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .stat-box {
        background: linear-gradient(135deg, #f0fdfa, #eff6ff);
        border: 1px solid #ccfbf1;
        border-radius: 10px;
        padding: 0.9rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.45rem;
        font-weight: 750;
        color: #0f766e;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #64748b;
    }
    .hint-box {
        background: #f8fafc;
        border-left: 3px solid #2563eb;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        color: #334155;
        font-size: 0.9rem;
        margin-bottom: 0.7rem;
    }
    .result-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.65rem;
    }
    .result-title {
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .meta-line {
        color: #64748b;
        font-size: 0.83rem;
        margin-bottom: 0.45rem;
    }
    .type-chip {
        display: inline-block;
        background: #0f766e22;
        color: #0f766e;
        border-radius: 8px;
        padding: 1px 8px;
        font-size: 0.72rem;
        font-weight: 650;
        margin: 2px 4px 2px 0;
    }
    .source-chip {
        display: inline-block;
        background: #2563eb22;
        color: #2563eb;
        border-radius: 8px;
        padding: 1px 8px;
        font-size: 0.72rem;
        font-weight: 650;
        margin-right: 4px;
    }
    .shortlist-chip {
        display: inline-block;
        background: #f59e0b22;
        color: #b45309;
        border-radius: 8px;
        padding: 2px 9px;
        font-size: 0.76rem;
        font-weight: 650;
        margin: 2px 4px 2px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_state():
    defaults = {
        "messages": [],
        "shortlist": [],
        "last_query": "",
        "top_n": 7,
        "top_k": 25,
        "language_filter": "Any",
        "test_type_filter": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


@st.cache_data(show_spinner=False)
def get_catalog():
    return load_catalog()


@st.cache_resource(show_spinner=False)
def get_recommender():
    return AssessmentRecommender(get_catalog(), use_dense=False, use_reranker=False)


def catalog_stats():
    catalog = get_catalog()
    languages = Counter(lang for item in catalog for lang in item.languages)
    test_types = Counter(t for item in catalog for t in item.test_types)
    job_levels = Counter(level for item in catalog for level in item.job_levels)
    return {
        "assessments": len(catalog),
        "languages": len(languages),
        "test_types": len(test_types),
        "job_levels": len(job_levels),
        "language_counts": languages,
        "test_type_counts": test_types,
        "job_level_counts": job_levels,
    }


def split_terms(text: str) -> List[str]:
    cleaned = re.sub(r"\b(and|or)\b", ",", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("|", ",").replace(";", ",")
    terms = [term.strip(" .:-") for term in cleaned.split(",")]
    return [term for term in terms if term]


def result_text(result: dict) -> str:
    meta = result["metadata"]
    parts = [
        meta.get("assessment_name", ""),
        result.get("document", ""),
        " ".join(meta.get("test_types", [])),
        " ".join(meta.get("languages", [])),
    ]
    return " ".join(parts).lower()


def contains_term(result: dict, term: str) -> bool:
    normalized = term.lower().strip()
    return bool(normalized and normalized in result_text(result))


def unique_results(results: Iterable[dict]) -> List[dict]:
    seen = set()
    unique = []
    for result in results:
        name = result["metadata"].get("assessment_name")
        if name and name not in seen:
            seen.add(name)
            unique.append(result)
    return unique


def extract_refinement_terms(prompt: str):
    drops = []
    adds = []

    replace_match = REPLACE_RE.search(prompt)
    if replace_match:
        drops.extend(split_terms(replace_match.group(1)))
        adds.extend(split_terms(replace_match.group(2)))

    for match in DROP_RE.finditer(prompt):
        drops.extend(split_terms(match.group(1)))
    for match in ADD_RE.finditer(prompt):
        adds.extend(split_terms(match.group(1)))

    return unique_terms(drops), unique_terms(adds)


def unique_terms(terms: List[str]) -> List[str]:
    seen = set()
    unique = []
    for term in terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            unique.append(term)
    return unique


def is_refinement(prompt: str) -> bool:
    drops, adds = extract_refinement_terms(prompt)
    return bool(drops or adds)


def is_small_talk(prompt: str) -> bool:
    return SMALL_TALK_RE.match(prompt) is not None


def active_filter_params() -> Optional[dict]:
    params = {}
    if st.session_state.language_filter != "Any":
        params["languages"] = [st.session_state.language_filter]
    if st.session_state.test_type_filter:
        params["test_types"] = st.session_state.test_type_filter
    return params or None


def answer_prompt(prompt: str) -> tuple[str, Optional[List[dict]], dict]:
    current = list(st.session_state.shortlist)

    if current and CONFIRMATION_RE.search(prompt):
        return "Confirmed. Here is the final shortlist.", current, {"kind": "confirmation"}

    if is_small_talk(prompt):
        if current:
            return (
                "Hi. I still have the current shortlist ready. Ask me to add or remove skills, or describe a new role.",
                None,
                {"kind": "small_talk"},
            )
        return (
            "Hi. Tell me the role, skills, job level, language, or constraints, and I will build an assessment shortlist.",
            None,
            {"kind": "small_talk"},
        )

    recommender = get_recommender()

    if current and is_refinement(prompt):
        drops, adds = extract_refinement_terms(prompt)
        refined = [
            result
            for result in current
            if not any(contains_term(result, term) for term in drops)
        ]

        added_results = []
        for term in adds:
            added_results.extend(
                recommender.query(
                    term,
                    filter_params=active_filter_params(),
                    top_k=st.session_state.top_k,
                    top_n=3,
                )
            )

        refined = unique_results([*refined, *added_results])[: st.session_state.top_n]
        st.session_state.shortlist = refined
        st.session_state.last_query = f"{st.session_state.last_query} {prompt}".strip()

        changes = []
        if drops:
            changes.append(f"removed {', '.join(drops)}")
        if adds:
            changes.append(f"added {', '.join(adds)}")
        reply = f"I updated the shortlist: {'; '.join(changes)}."
        return reply, refined, {"kind": "refinement", "drops": drops, "adds": adds}

    results = recommender.query(
        prompt,
        filter_params=active_filter_params(),
        top_k=st.session_state.top_k,
        top_n=st.session_state.top_n,
    )
    st.session_state.shortlist = results
    st.session_state.last_query = prompt
    return (
        "Here is the best matching assessment shortlist from the catalog.",
        results,
        {"kind": "recommendation"},
    )


def test_type_chips(test_types: List[str]) -> str:
    chips = []
    for test_type in test_types:
        label = TEST_TYPE_LABELS.get(test_type, test_type)
        chips.append(f'<span class="type-chip">{test_type}: {label}</span>')
    return " ".join(chips) if chips else '<span class="type-chip">Not specified</span>'


def compact_list(values: List[str], limit: int = 4) -> str:
    if not values:
        return "-"
    if len(values) <= limit:
        return ", ".join(values)
    return f"{', '.join(values[:limit])} (+{len(values) - limit} more)"


def render_results(results: Optional[List[dict]]):
    if results is None:
        return
    if not results:
        st.info("No matching assessments found.")
        return

    st.markdown(format_table(results), unsafe_allow_html=False)

    with st.expander(f"View {len(results)} recommendation details", expanded=False):
        for index, result in enumerate(results, 1):
            meta = result["metadata"]
            name = meta.get("assessment_name", "Unknown")
            source = result.get("source", "retrieval")
            score = result.get("rrf_score", result.get("score", 0.0))
            st.markdown(
                f"""
<div class="result-box">
    <div class="result-title">{index}. {name}</div>
    <div class="meta-line">
        <span class="source-chip">{source}</span>
        Score: {score:.4f} · Duration: {meta.get("assessment_length") or "-"} · Remote: {meta.get("remote_testing")}
    </div>
    <div>{test_type_chips(meta.get("test_types", []))}</div>
    <div class="meta-line">Languages: {compact_list(meta.get("languages", []))}</div>
    <div class="meta-line">Job levels: {compact_list(meta.get("job_levels", []))}</div>
    <div class="meta-line"><a href="{meta.get("url", "#")}" target="_blank">Open product page</a></div>
</div>
""",
                unsafe_allow_html=True,
            )


def append_turn(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    reply, results, meta = answer_prompt(prompt)
    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "results": results, "meta": meta}
    )


def reset_conversation():
    st.session_state.messages = []
    st.session_state.shortlist = []
    st.session_state.last_query = ""


def sidebar():
    stats = catalog_stats()
    with st.sidebar:
        st.markdown("## Configuration")

        st.session_state.top_n = st.slider(
            "Shortlist size",
            min_value=3,
            max_value=10,
            value=st.session_state.top_n,
        )
        st.session_state.top_k = st.slider(
            "Retrieve Top-K",
            min_value=10,
            max_value=50,
            value=st.session_state.top_k,
            step=5,
        )

        st.divider()

        language_options = ["Any"] + sorted(stats["language_counts"].keys())
        st.session_state.language_filter = st.selectbox(
            "Language",
            language_options,
            index=language_options.index(st.session_state.language_filter)
            if st.session_state.language_filter in language_options
            else 0,
        )

        type_options = sorted(stats["test_type_counts"].keys())
        st.session_state.test_type_filter = st.multiselect(
            "Test types",
            type_options,
            default=st.session_state.test_type_filter,
            format_func=lambda t: f"{t} - {TEST_TYPE_LABELS.get(t, t)}",
        )

        st.divider()
        st.markdown("## Active Shortlist")
        if st.session_state.shortlist:
            for result in st.session_state.shortlist[:10]:
                name = result["metadata"].get("assessment_name", "Unknown")
                st.markdown(f'<span class="shortlist-chip">{name}</span>', unsafe_allow_html=True)
        else:
            st.caption("No shortlist yet.")

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            reset_conversation()
            st.rerun()
        if st.button("Reset everything", use_container_width=True):
            reset_conversation()
            st.session_state.language_filter = "Any"
            st.session_state.test_type_filter = []
            st.rerun()


def render_stats():
    stats = catalog_stats()
    cols = st.columns(4)
    items = [
        ("Assessments", stats["assessments"]),
        ("Languages", stats["languages"]),
        ("Test Types", stats["test_types"]),
        ("Job Levels", stats["job_levels"]),
    ]
    for col, (label, value) in zip(cols, items):
        col.markdown(
            f"""
<div class="stat-box">
    <div class="stat-number">{value}</div>
    <div class="stat-label">{label}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_empty_state():
    st.markdown(
        """
<div class="hint-box">
Ask for a role, skills, job level, language, or constraints. Follow up later with changes like
<strong>drop REST and add AWS and Docker</strong>.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("**Try asking:**")
    suggestions = [
        "Senior backend engineer with Core Java, Spring, SQL, AWS, and Docker",
        "Healthcare administrator with HIPAA, medical terminology, and Microsoft Word",
        "Graduate analyst role with numerical reasoning and situational judgement",
        "Customer service role with phone simulation and written English",
    ]
    cols = st.columns(2)
    for index, suggestion in enumerate(suggestions):
        if cols[index % 2].button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            append_turn(suggestion)
            st.rerun()


def render_chat_tab():
    if not st.session_state.messages:
        render_empty_state()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "results" in message:
                render_results(message["results"])

    prompt = st.chat_input("Ask for assessments or refine the current shortlist")
    if prompt:
        append_turn(prompt)
        st.rerun()


def render_catalog_tab():
    stats = catalog_stats()
    st.markdown("### Catalog Overview")

    col_types, col_langs = st.columns(2)
    with col_types:
        st.markdown("#### Test Type Coverage")
        type_rows = [
            {
                "Type": key,
                "Meaning": TEST_TYPE_LABELS.get(key, key),
                "Assessments": count,
            }
            for key, count in stats["test_type_counts"].most_common()
        ]
        st.dataframe(pd.DataFrame(type_rows), use_container_width=True, hide_index=True)

    with col_langs:
        st.markdown("#### Top Languages")
        language_rows = [
            {"Language": key, "Assessments": count}
            for key, count in stats["language_counts"].most_common(15)
        ]
        st.dataframe(pd.DataFrame(language_rows), use_container_width=True, hide_index=True)

    st.markdown("#### Catalog Search")
    search = st.text_input("Search assessment names/descriptions", placeholder="Java, OPQ, HIPAA, Excel...")
    if search:
        query = search.lower()
        rows = []
        for item in get_catalog():
            text = f"{item.name} {item.description}".lower()
            if query in text:
                rows.append(
                    {
                        "Name": item.name,
                        "Test Type": ", ".join(item.test_types),
                        "Duration": item.assessment_length or "-",
                        "Languages": compact_list(item.languages),
                        "URL": item.url,
                    }
                )
        st.dataframe(pd.DataFrame(rows[:50]), use_container_width=True, hide_index=True)


def main():
    init_state()
    sidebar()

    st.markdown('<div class="main-header">AssessQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Catalog-grounded assessment recommendations with shortlist memory</div>',
        unsafe_allow_html=True,
    )

    render_stats()
    st.markdown("---")

    tab_chat, tab_catalog = st.tabs(["Chat", "Catalog"])
    with tab_chat:
        render_chat_tab()
    with tab_catalog:
        render_catalog_tab()


if __name__ == "__main__":
    main()
