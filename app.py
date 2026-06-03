import re
import uuid
from collections import Counter
from typing import Iterable, List, Optional

import pandas as pd
import streamlit as st

from agent import (
    AssessmentRecommender,
    ConversationalRecommender,
    TEST_TYPE_LABELS,
    format_table,
    load_catalog,
)

# ── regex helpers (still used for lightweight shortlist ops) ──────────────────
CONFIRMATION_RE = re.compile(
    r"\b(perfect|looks good|that covers it|lock|locking|final|done|yes|correct|great)\b",
    re.IGNORECASE,
)
SMALL_TALK_RE = re.compile(
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|thanks|thank\s+you|help|what\s+can\s+you\s+do)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

# ── page config ───────────────────────────────────────────────────────────────
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
        font-size: 2rem; font-weight: 750;
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle { color: #64748b; font-size: 0.95rem; margin-bottom: 1.2rem; }
    .stat-box {
        background: linear-gradient(135deg, #f0fdfa, #eff6ff);
        border: 1px solid #ccfbf1; border-radius: 10px;
        padding: 0.9rem; text-align: center;
    }
    .stat-number { font-size: 1.45rem; font-weight: 750; color: #0f766e; }
    .stat-label  { font-size: 0.78rem; color: #64748b; }
    .hint-box {
        background: #f8fafc; border-left: 3px solid #2563eb;
        padding: 0.75rem 1rem; border-radius: 0 8px 8px 0;
        color: #334155; font-size: 0.9rem; margin-bottom: 0.7rem;
    }
    .result-box {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 0.9rem 1rem; margin-bottom: 0.65rem;
    }
    .result-title  { font-weight: 700; color: #0f172a; margin-bottom: 0.25rem; }
    .meta-line     { color: #64748b; font-size: 0.83rem; margin-bottom: 0.45rem; }
    .type-chip {
        display: inline-block; background: #0f766e22; color: #0f766e;
        border-radius: 8px; padding: 1px 8px; font-size: 0.72rem;
        font-weight: 650; margin: 2px 4px 2px 0;
    }
    .source-chip {
        display: inline-block; background: #2563eb22; color: #2563eb;
        border-radius: 8px; padding: 1px 8px; font-size: 0.72rem;
        font-weight: 650; margin-right: 4px;
    }
    .shortlist-chip {
        display: inline-block; background: #f59e0b22; color: #b45309;
        border-radius: 8px; padding: 2px 9px; font-size: 0.76rem;
        font-weight: 650; margin: 2px 4px 2px 0;
    }
    .memory-badge {
        display: inline-block; background: #dcfce7; color: #166534;
        border-radius: 6px; padding: 2px 8px; font-size: 0.72rem;
        font-weight: 600; margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── session state init ────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "messages": [],          # UI chat history  [{role, content, results, meta}]
        "shortlist": [],         # current assessment shortlist
        "last_query": "",
        "top_n": 7,
        "top_k": 25,
        "language_filter": "Any",
        "test_type_filter": [],
        # Unique session ID — used as the key for LangChain message history.
        # One UUID per browser tab; survives reruns, resets on Clear.
        "session_id": str(uuid.uuid4()),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ── cached resources ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_catalog():
    return load_catalog()


@st.cache_resource(show_spinner=False)
def get_recommender():
    return AssessmentRecommender(get_catalog(), use_dense=False, use_reranker=False)


@st.cache_resource(show_spinner=False)
def get_conversational_recommender():
    """
    Single ConversationalRecommender shared across reruns.
    The per-session conversation history is NOT stored here — it lives in
    st.session_state via StreamlitChatMessageHistory, keyed by session_id.
    """
    return ConversationalRecommender(get_recommender())


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


# ── filter helpers ────────────────────────────────────────────────────────────

def active_filter_params() -> Optional[dict]:
    params = {}
    if st.session_state.language_filter != "Any":
        params["languages"] = [st.session_state.language_filter]
    if st.session_state.test_type_filter:
        params["test_types"] = st.session_state.test_type_filter
    return params or None


# ── shortlist helpers (for UI-side drop/add still applied to saved shortlist) ─

def unique_results(results: Iterable[dict]) -> List[dict]:
    seen = set()
    unique = []
    for result in results:
        name = result["metadata"].get("assessment_name")
        if name and name not in seen:
            seen.add(name)
            unique.append(result)
    return unique


def contains_term(result: dict, term: str) -> bool:
    meta = result["metadata"]
    parts = [
        meta.get("assessment_name", ""),
        result.get("document", ""),
        " ".join(meta.get("test_types", [])),
        " ".join(meta.get("languages", [])),
    ]
    text = " ".join(parts).lower()
    return bool(term.lower().strip() in text)


def apply_shortlist_refine(drops: List[str], adds: List[str], new_results: Optional[List[dict]]) -> List[dict]:
    """Apply drop/add instructions to the current shortlist."""
    current = list(st.session_state.shortlist)

    if drops:
        current = [r for r in current if not any(contains_term(r, t) for t in drops)]

    if new_results:
        current = unique_results([*current, *new_results])[: st.session_state.top_n]

    return current


# ── main conversation handler ─────────────────────────────────────────────────

def answer_prompt(prompt: str) -> tuple[str, Optional[List[dict]], dict]:
    """
    Route the user message through the LLM-powered ConversationalRecommender.
    Returns (reply_text, results_or_None, meta_dict).
    """
    conv = get_conversational_recommender()

    response = conv.chat(
        user_input=prompt,
        session_id=st.session_state.session_id,   
        filter_params=active_filter_params(),
        top_k=st.session_state.top_k,
        top_n=st.session_state.top_n,
    )

    action  = response["action"]
    reply   = response["reply"]
    results = response.get("results")
    drops   = response.get("drops", [])
    adds    = response.get("adds", [])

    if action == "recommend" and isinstance(results, list):
        st.session_state.shortlist = results

    elif action == "refine":
        # LLM identified drops/adds; new retrieval results may also come back
        updated = apply_shortlist_refine(drops, adds, results)
        st.session_state.shortlist = updated
        results = updated

    elif action == "confirm":
        results = list(st.session_state.shortlist)

    return reply, results, {"action": action, "drops": drops, "adds": adds}


# ── rendering ─────────────────────────────────────────────────────────────────

def test_type_chips(test_types: List[str]) -> str:
    chips = []
    for tt in test_types:
        label = TEST_TYPE_LABELS.get(tt, tt)
        chips.append(f'<span class="type-chip">{tt}: {label}</span>')
    return " ".join(chips) if chips else '<span class="type-chip">Not specified</span>'


def compact_list(values: List[str], limit: int = 4) -> str:
    if not values:
        return "-"
    if len(values) <= limit:
        return ", ".join(values)
    return f"{', '.join(values[:limit])} (+{len(values) - limit} more)"


def render_results(results):
    if results is None:
        return
    if isinstance(results, str):
        # Legacy fallback: results was accidentally a raw LLM string
        st.markdown(results)
        return
    if not results:
        st.info("No matching assessments found.")
        return

    st.markdown(format_table(results), unsafe_allow_html=False)

    with st.expander(f"View {len(results)} recommendation details", expanded=False):
        for index, result in enumerate(results, 1):
            meta = result["metadata"]
            name  = meta.get("assessment_name", "Unknown")
            source = result.get("source", "retrieval")
            score  = result.get("rrf_score", result.get("score", 0.0))
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
    """Clear UI history, shortlist, AND the LangChain message history in session_state."""
    st.session_state.messages = []
    st.session_state.shortlist = []
    st.session_state.last_query = ""
    new_sid = str(uuid.uuid4())
    st.session_state.session_id = new_sid
    lc_key = f"lc_history_{new_sid}"
    if lc_key in st.session_state:
        del st.session_state[lc_key]


# ── sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    stats = catalog_stats()
    with st.sidebar:
        st.markdown("## Configuration")

        # Memory status badge
        history_key = f"lc_history_{st.session_state.session_id}"
        history_msgs = st.session_state.get(history_key, [])
        turn_count = len(history_msgs) // 2
        st.markdown(
            f'<span class="memory-badge"> Memory: {turn_count} turn{"s" if turn_count != 1 else ""} stored</span>',
            unsafe_allow_html=True,
        )

        st.session_state.top_n = st.slider("Shortlist size", 3, 10, st.session_state.top_n)
        st.session_state.top_k = st.slider("Retrieve Top-K", 10, 50, st.session_state.top_k, step=5)

        st.divider()

        language_options = ["Any"] + sorted(stats["language_counts"].keys())
        st.session_state.language_filter = st.selectbox(
            "Language",
            language_options,
            index=language_options.index(st.session_state.language_filter)
            if st.session_state.language_filter in language_options else 0,
        )
        type_options = sorted(stats["test_type_counts"].keys())
        st.session_state.test_type_filter = st.multiselect(
            "Test types", type_options,
            default=st.session_state.test_type_filter,
            format_func=lambda t: f"{t} - {TEST_TYPE_LABELS.get(t, t)}",
        )

        st.divider()
        st.markdown("## Active Shortlist")
        if st.session_state.shortlist and isinstance(st.session_state.shortlist, list):
            for result in st.session_state.shortlist[:10]:
                if not isinstance(result, dict):
                    continue
                name = result.get("metadata", {}).get("assessment_name", "Unknown")
                st.markdown(f'<span class="shortlist-chip">{name}</span>', unsafe_allow_html=True)
        else:
            st.caption("No shortlist yet.")

        st.divider()
        if st.button("Clear chat", width='stretch'):
            reset_conversation()
            st.rerun()
        if st.button("Reset everything", width='stretch'):
            reset_conversation()
            st.session_state.language_filter = "Any"
            st.session_state.test_type_filter = []
            st.rerun()

        with st.expander("🔍 Session debug", expanded=False):
            st.caption(f"session_id: `{st.session_state.session_id[:8]}…`")
            st.caption(f"LangChain history key: `lc_history_{st.session_state.session_id[:8]}…`")
            if history_msgs:
                for msg in history_msgs[-6:]:   # show last 3 turns
                    role = getattr(msg, "type", "?")
                    content = getattr(msg, "content", str(msg))
                    st.caption(f"**{role}**: {content[:80]}…")


# ── stat cards ────────────────────────────────────────────────────────────────

def render_stats():
    stats = catalog_stats()
    cols = st.columns(4)
    items = [
        ("Assessments", stats["assessments"]),
        ("Languages",   stats["languages"]),
        ("Test Types",  stats["test_types"]),
        ("Job Levels",  stats["job_levels"]),
    ]
    for col, (label, value) in zip(cols, items):
        col.markdown(
            f'<div class="stat-box"><div class="stat-number">{value}</div>'
            f'<div class="stat-label">{label}</div></div>',
            unsafe_allow_html=True,
        )


# ── empty state ───────────────────────────────────────────────────────────────

def render_empty_state():
    st.markdown(
        """
<div class="hint-box">
Describe the role or paste a job description. The assistant will ask a couple of clarifying
questions, then build a shortlist. Follow up with changes like
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
        if cols[index % 2].button(suggestion, key=f"suggestion_{index}", width='stretch'):
            append_turn(suggestion)
            st.rerun()


# ── chat tab ──────────────────────────────────────────────────────────────────

def render_chat_tab():
    if not st.session_state.messages:
        render_empty_state()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("results"):
                render_results(message["results"])

    prompt = st.chat_input("Describe the role, or refine the current shortlist")
    if prompt:
        append_turn(prompt)
        st.rerun()


# ── catalog tab ───────────────────────────────────────────────────────────────

def render_catalog_tab():
    stats = catalog_stats()
    st.markdown("### Catalog Overview")
    col_types, col_langs = st.columns(2)
    with col_types:
        st.markdown("#### Test Type Coverage")
        st.dataframe(pd.DataFrame([
            {"Type": k, "Meaning": TEST_TYPE_LABELS.get(k, k), "Assessments": v}
            for k, v in stats["test_type_counts"].most_common()
        ]), width='stretch', hide_index=True)
    with col_langs:
        st.markdown("#### Top Languages")
        st.dataframe(pd.DataFrame([
            {"Language": k, "Assessments": v}
            for k, v in stats["language_counts"].most_common(15)
        ]), width='stretch', hide_index=True)

    st.markdown("#### Catalog Search")
    search = st.text_input("Search assessment names/descriptions", placeholder="Java, OPQ, HIPAA, Excel…")
    if search:
        query = search.lower()
        rows = []
        for item in get_catalog():
            if query in f"{item.name} {item.description}".lower():
                rows.append({
                    "Name": item.name,
                    "Test Type": ", ".join(item.test_types),
                    "Duration": item.assessment_length or "-",
                    "Languages": compact_list(item.languages),
                    "URL": item.url,
                })
        st.dataframe(pd.DataFrame(rows[:50]), width='stretch', hide_index=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    init_state()
    sidebar()

    st.markdown('<div class="main-header">AssessQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">LLM-powered assessment recommendations with full conversation memory</div>',
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