# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streamlit web UI for the Drug Interaction Monitor Agent."""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import threading
import time
import warnings
from pathlib import Path

# Suppress noisy third-party warnings that don't affect functionality
warnings.filterwarnings("ignore", message=".*EXPERIMENTAL.*")
warnings.filterwarnings("ignore", message=".*collector endpoint protocol.*")
warnings.filterwarnings("ignore", message=".*non-text parts.*")

logging.getLogger("opentelemetry").setLevel(logging.ERROR)
logging.getLogger("openinference").setLevel(logging.ERROR)

_agent_dir = str(Path(__file__).resolve().parent)
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import streamlit as st

st.set_page_config(
    page_title="Drug Interaction Monitor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from opentelemetry import trace as otel_trace
from opentelemetry.trace import StatusCode

from instrumentation import setup_tracing
from drug_interaction_agent.tools import (
    _PIPELINE_CACHE,
    _resolve_single_rxcui,
    apply_self_improvement,
    check_all_interactions,
    evaluate_report_quality,
    get_fda_enrichment,
    resolve_all_medications,
    synthesize_safety_report,
)
from phoenix_client import ensure_prompt_synced, get_historical_weaknesses, phoenix_mcp_session


@st.cache_resource(show_spinner="Fetching top drugs from FDA database...")
def _fetch_top_drugs() -> tuple[list[str], list[str]]:
    """Fetch generic drug names from OpenFDA NDC database.

    Returns (full_list, preresolved_subset):
    - full_list: up to 500 names for the dropdown typeahead
    - preresolved_subset: top 20 names to pre-resolve via RxNorm at startup

    Fetching 500 covers virtually all common drugs (warfarin, clopidogrel, etc.)
    while pre-resolving only 20 keeps startup fast for Cloud Run cold starts.
    """
    import httpx
    fallback = sorted([
        "acetaminophen", "albuterol", "amlodipine", "amoxicillin",
        "aspirin", "atorvastatin", "clopidogrel", "gabapentin",
        "hydrochlorothiazide", "ibuprofen", "levothyroxine", "lisinopril",
        "metformin", "metoprolol", "omeprazole", "sertraline",
        "simvastatin", "tramadol", "warfarin",
    ])
    try:
        resp = httpx.get(
            "https://api.fda.gov/drug/ndc.json",
            params={"count": "generic_name.exact", "limit": "1000"},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        seen: set[str] = set()
        drugs: list[str] = []
        for r in results:
            name = r.get("term", "").strip().lower()
            if name and ";" not in name and 3 <= len(name) <= 40 and name not in seen:
                seen.add(name)
                drugs.append(name)
            if len(drugs) >= 500:
                break
        full = sorted(drugs)
        return full, full[:20]
    except Exception:
        return fallback, fallback

SECTION_ORDER = [
    "MEDICATION SUMMARY",
    "INTERACTION ALERTS",
    "FDA LABEL WARNINGS",
    "REAL-WORLD ADVERSE EVENTS",
    "RECOMMENDATIONS",
]

SECTION_ICONS = {
    "MEDICATION SUMMARY": "💊",
    "INTERACTION ALERTS": "⚠️",
    "FDA LABEL WARNINGS": "📋",
    "REAL-WORLD ADVERSE EVENTS": "📊",
    "RECOMMENDATIONS": "✅",
}

# ---------------------------------------------------------------------------
# One-time startup: tracing (background thread) + RxNorm pre-resolution
# ---------------------------------------------------------------------------

@st.cache_resource
def _init_services() -> threading.Event:
    ready = threading.Event()
    def _bg():
        setup_tracing()
        ready.set()
    threading.Thread(target=_bg, daemon=True).start()
    return ready


@st.cache_resource(show_spinner="Pre-resolving drugs via RxNorm...")
def _preresolved_drugs(drug_list: tuple[str, ...]) -> dict[str, dict]:
    """Resolve the FDA-fetched drug list to RxCUI once at startup. Cached permanently.

    Returns a dict keyed by lowercase drug name → Drug model_dump().
    Any drug the user selects from the dropdown uses this cache directly,
    skipping the RxNorm API call entirely during the pipeline run.
    """
    async def _resolve_all():
        results = await asyncio.gather(
            *[_resolve_single_rxcui(name) for name in drug_list]
        )
        return {drug.name.lower(): drug.model_dump() for drug in results}

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_resolve_all())
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Pipeline - direct Python pipeline (no ADK overhead)
# ---------------------------------------------------------------------------

async def _run_pipeline(selected_drugs: list[str], pre_resolved: dict, status=None) -> dict:
    async with phoenix_mcp_session():
        await ensure_prompt_synced()
        historical_weaknesses_display, historical_weaknesses_prompt, historical_runs = await get_historical_weaknesses()
        _PIPELINE_CACHE.clear()
        _PIPELINE_CACHE["historical_weaknesses"] = historical_weaknesses_prompt
        _PIPELINE_CACHE["historical_runs"] = historical_runs
        _PIPELINE_CACHE["pre_resolved"] = pre_resolved
        _PIPELINE_CACHE["selected_drugs"] = list(selected_drugs)

        medication_text = "I take " + ", ".join(selected_drugs)
        tracer = otel_trace.get_tracer("drug_interaction_agent")

        async def _span(name: str, coro, input_val: str = ""):
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("openinference.span.kind", "TOOL")
                span.set_attribute("tool.name", name)
                if input_val:
                    span.set_attribute("input.value", input_val)
                result = await coro
                if result is not None:
                    out = str(result)
                    span.set_attribute("output.value", out)
                    stripped = out.lstrip()
                    mime = "application/json" if stripped and stripped[0] in ("{", "[") else "text/plain"
                    span.set_attribute("output.mime_type", mime)
                span.set_status(StatusCode.OK)
                return result

        with tracer.start_as_current_span("drug_interaction_pipeline") as root:
            root.set_attribute("openinference.span.kind", "CHAIN")
            root.set_attribute("input.value", medication_text)

            if status:
                status.update(label="Resolving medications via RxNorm...")
            drugs_json = await _span("resolve_all_medications",
                resolve_all_medications(medication_text),
                input_val=medication_text)

            if status:
                status.update(label="Checking interactions and FDA data in parallel...")
            interactions_json, _ = await asyncio.gather(
                _span("check_all_interactions", check_all_interactions(drugs_json), input_val=drugs_json),
                _span("get_fda_enrichment", get_fda_enrichment(drugs_json), input_val=drugs_json),
            )

            if status:
                status.update(label="Synthesizing safety report...")
            report = await _span("synthesize_safety_report",
                synthesize_safety_report(drugs_json, interactions_json),
                input_val=drugs_json)

            if status:
                status.update(label="Evaluating report quality...")
            eval_json = await _span("evaluate_report_quality",
                evaluate_report_quality(report),
                input_val=report)

            if status:
                status.update(label="Applying self-improvement...")
            output = await _span("apply_self_improvement",
                apply_self_improvement(report),
                input_val=eval_json)

            root.set_attribute("output.value", _PIPELINE_CACHE.get("report", output or ""))
            root.set_status(StatusCode.OK)

        final = _PIPELINE_CACHE.get("report", output or report)

    return {
        "report": final,
        "eval_scores": _PIPELINE_CACHE.get("eval_scores", []),
        "improvement_log": _PIPELINE_CACHE.get("improvement_log", []),
        "drugs": _PIPELINE_CACHE.get("drugs", []),
        "interactions": _PIPELINE_CACHE.get("interactions", []),
        "judge_prompt_version": _PIPELINE_CACHE.get("judge_prompt_version", "built-in"),
        "historical_runs": _PIPELINE_CACHE.get("historical_runs", historical_runs),
        "historical_weaknesses": historical_weaknesses_display,
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_icon(score: float) -> str:
    if score > 0.75:
        return "🟢"
    if score > 0.4:
        return "🟡"
    return "🔴"


def _parse_sections(report_text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    pattern = re.compile(
        r"\d+\.\s*\**\s*(MEDICATION SUMMARY|INTERACTION ALERTS|FDA LABEL WARNINGS"
        r"|REAL-WORLD ADVERSE EVENTS|RECOMMENDATIONS)\**",
        re.IGNORECASE,
    )
    parts = pattern.split(report_text)
    for i in range(1, len(parts) - 1, 2):
        name = parts[i].strip().upper()
        content = parts[i + 1].strip()
        content = re.sub(r"\n---.*", "", content, flags=re.DOTALL).strip()
        # Strip the embedded score table - it's shown in the sidebar instead
        content = re.sub(r"\n*Per-section confidence scores.*", "", content, flags=re.DOTALL).strip()
        sections[name] = content
    return sections


def _highlight_severity(text: str) -> str:
    text = re.sub(r"\[MAJOR\]", "🔴 **MAJOR**", text, flags=re.IGNORECASE)
    text = re.sub(r"\[MODERATE\]", "🟠 **MODERATE**", text, flags=re.IGNORECASE)
    text = re.sub(r"\[MINOR\]", "🟡 **MINOR**", text, flags=re.IGNORECASE)
    return text

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(results: dict) -> None:
    eval_scores = results.get("eval_scores", [])
    improvement_map = {e["section"]: e for e in results.get("improvement_log", [])}

    with st.sidebar:
        st.markdown("<h1 style='margin-top:-1rem'>Analysis Results</h1>", unsafe_allow_html=True)

        drugs = results.get("drugs", [])
        interactions = results.get("interactions", [])
        resolved = sum(1 for d in drugs if d.get("resolved"))

        col1, col2 = st.columns(2)
        col1.metric("Drugs", f"{resolved}/{len(drugs)}", help="Resolved via RxNorm / Total")
        col2.metric("Interactions", len(interactions))

        st.divider()

        if eval_scores:
            st.subheader("Section Quality")
            avg = sum(ev.get("overall", 0.5) for ev in eval_scores) / len(eval_scores)
            st.metric("Overall Score", f"{avg:.2f} / 1.00")
            st.progress(avg)
            st.caption(" ")

            for ev in eval_scores:
                section = ev["section"]
                score = ev.get("overall", 0.5)
                short = section.replace("REAL-WORLD ADVERSE EVENTS", "REAL-WORLD EVENTS")
                imp = improvement_map.get(section)
                if imp and imp["delta"] != 0:
                    st.metric(
                        label=f"{_score_icon(score)} {short}",
                        value=f"{score:.2f}",
                        delta=f"{imp['delta']:+.2f} (was {imp['before']:.2f})",
                    )
                else:
                    st.metric(label=f"{_score_icon(score)} {short}", value=f"{score:.2f}")

        st.divider()

        if eval_scores:
            st.subheader("Dimension Breakdown")
            for dim in ("accuracy", "caution", "clarity", "citation"):
                avg_dim = sum(ev.get(dim, 0.5) for ev in eval_scores) / len(eval_scores)
                st.caption(f"{dim.capitalize()}: {avg_dim:.2f}")
                st.progress(avg_dim)

        st.divider()

        st.subheader("Pipeline Info")
        st.caption(f"**Historical Runs:** {results.get('historical_runs', 0)}")
        st.caption(f"**Judge Criteria:** {results.get('judge_prompt_version', 'built-in')}")
        st.caption("**Tools:** resolve -> check_interactions || fda_enrichment -> synthesize -> evaluate -> improve")
        st.caption("**Data:** NIH RxNorm · FDA Labels · FAERS · Gemini · Phoenix")

# ---------------------------------------------------------------------------
# Report view
# ---------------------------------------------------------------------------

def render_report(results: dict) -> None:
    report = results.get("report", "")
    sections = _parse_sections(report)

    st.subheader("Drug Interaction Safety Report")

    for section_name in SECTION_ORDER:
        content = sections.get(section_name, "")
        if not content:
            continue

        icon = SECTION_ICONS.get(section_name, "📌")
        label = f"{icon} {section_name.title().replace('Fda', 'FDA')}"

        expanded = section_name in ("INTERACTION ALERTS", "MEDICATION SUMMARY")
        with st.expander(label, expanded=expanded):
            if section_name == "INTERACTION ALERTS":
                st.markdown(_highlight_severity(content))
            else:
                st.markdown(content)

    disclaimer = re.search(r"DISCLAIMER:.*", report, re.IGNORECASE | re.DOTALL)
    if disclaimer:
        st.caption("---")
        st.caption(disclaimer.group(0)[:500])

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    tracing_ready = _init_services()
    drug_list, preresolved_subset = _fetch_top_drugs()
    pre_resolved = _preresolved_drugs(tuple(preresolved_subset))
    has_results = "results" in st.session_state

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("<style>div.block-container{padding-top:1.5rem}</style>", unsafe_allow_html=True)
    st.title("💊 Drug Interaction Monitor")
    if not has_results:
        st.caption(
            "AI-powered safety analysis · NIH RxNorm · FDA Labels · FAERS · "
            "Gemini 2.5 Flash · Arize Phoenix Observability"
        )

    # ── Drug selector ────────────────────────────────────────────────────────
    with st.container():
        if not has_results:
            st.subheader("Select Medications")
            st.caption(
                "Search from FDA-sourced drugs, or type any drug name - "
                "known or unknown - and press **Enter** to add it."
            )
        else:
            st.write("")

        selected_drugs: list[str] = st.multiselect(
            label="Medications",
            options=drug_list,
            placeholder="Type to search - e.g. warfarin, metformin, lisinopril...",
            accept_new_options=True,
            label_visibility="collapsed",
        )

        if selected_drugs:
            known = [d for d in selected_drugs if d.lower() in pre_resolved and pre_resolved[d.lower()].get("resolved")]
            unresolved_known = [d for d in selected_drugs if d.lower() in pre_resolved and not pre_resolved[d.lower()].get("resolved")]
            custom = [d for d in selected_drugs if d.lower() not in pre_resolved]
            status_parts = []
            if known:
                status_parts.append(f"✅ Pre-resolved: {', '.join(known)}")
            if unresolved_known:
                status_parts.append(f"⚠️ Not in RxNorm: {', '.join(unresolved_known)}")
            if custom:
                status_parts.append(f"🔍 Will resolve live: {', '.join(custom)}")
            st.caption("  ·  ".join(status_parts))

        initializing = not tracing_ready.is_set()
        run_clicked = st.button(
            "🔍  Check Drug Interactions",
            type="primary",
            disabled=len(selected_drugs) < 1 or initializing,
            use_container_width=True,
        )
        if initializing:
            st.caption("⏳ Initializing tracing... please wait a moment.")
            time.sleep(1.0 if selected_drugs else 0.5)
            st.rerun()

    if run_clicked and selected_drugs:
        cache_key = "result_" + "_".join(sorted(d.lower() for d in selected_drugs))
        if cache_key in st.session_state:
            st.session_state.results = st.session_state[cache_key]
        else:
            try:
                with st.status("Starting analysis...", expanded=True) as status:
                    results = asyncio.run(
                        _run_pipeline(selected_drugs, pre_resolved, status=status)
                    )
                    status.update(label="Analysis complete!", state="complete")
                st.session_state.results = results
                st.session_state[cache_key] = results
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                return

    if "results" in st.session_state:
        render_sidebar(st.session_state.results)
        st.divider()
        render_report(st.session_state.results)


if __name__ == "__main__":
    main()
