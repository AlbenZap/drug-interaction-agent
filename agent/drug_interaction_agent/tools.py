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

"""All tool functions for the Drug Interaction Agent.

Six tools implement the full pipeline:
  1. resolve_all_medications   — NLU extraction + RxNorm resolution
  2. check_all_interactions    — NIH Drug Interaction API for every drug pair
  3. get_fda_enrichment        — FDA label warnings + FAERS adverse events
  4. synthesize_safety_report  — Gemini synthesis of the 5-section report
  5. evaluate_report_quality   — LLM-as-a-Judge, judge criteria from Phoenix Prompt
                                  Management, scores annotated on Phoenix traces
  6. apply_self_improvement    — rerun low-confidence sections, write history to
                                  Phoenix dataset for cross-run learning
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from itertools import combinations
from typing import Any

import httpx
from opentelemetry import trace
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from drug_interaction_agent.models import (
    AdverseEvent,
    Drug,
    DrugLabel,
    Interaction,
    SectionEval,
)
from drug_interaction_agent.prompts import (
    IMPROVE_PROMPT,
    JUDGE_BATCH_PROMPT,
    SYNTHESIS_PROMPT,
)

logger = logging.getLogger(__name__)

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_BASE = "https://api.fda.gov/drug"

SECTION_NAMES = [
    "MEDICATION SUMMARY",
    "INTERACTION ALERTS",
    "FDA LABEL WARNINGS",
    "REAL-WORLD ADVERSE EVENTS",
    "RECOMMENDATIONS",
]

# Module-level cache: stores large data between tool calls so it never has to
# pass through the LLM as a function call argument (avoids MALFORMED_FUNCTION_CALL).
_PIPELINE_CACHE: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Shared HTTP helper with retry
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True,
)
async def _http_get(url: str, params: dict[str, Any] | None = None) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, params=params or {})
        if resp.status_code == 404:
            return {}  # no results found — not a server error
        resp.raise_for_status()
        return resp.json()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    reraise=True,
)
async def _http_get_raw_url(url: str) -> dict:
    """GET a fully-constructed URL without httpx param encoding."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Gemini helper
# ---------------------------------------------------------------------------


def _gemini_client():
    import google.genai as genai  # lazy import — keeps startup fast

    return genai.Client()


def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


async def _generate(prompt: str, thinking: bool = True) -> str:
    """Call Gemini. Pass thinking=False for structured JSON tasks (scoring, extraction)
    to skip the reasoning budget — saves 20-30s on gemini-2.5-flash."""
    import google.genai as genai
    client = _gemini_client()
    config = None
    if not thinking and "2.5" in _model_name():
        config = genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0)
        )
    for attempt in range(4):
        try:
            response = await client.aio.models.generate_content(
                model=_model_name(),
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                wait = 15 * (2 ** attempt)
                logger.warning("Gemini 429 rate limit — retrying in %ds (attempt %d/4)", wait, attempt + 1)
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini rate limit persisted after 4 retries")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_text(text: str) -> str:
    """Strip backslashes and control chars that corrupt JSON serialization."""
    cleaned = re.sub(r"\\(?![\"\\\/bfnrtu])", " ", str(text))
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", cleaned)
    return cleaned.strip()


async def _resolve_single_rxcui(drug_name: str) -> Drug:
    """Call RxNorm to resolve one drug name to an RxCUI."""
    try:
        data = await _http_get(
            f"{RXNORM_BASE}/rxcui.json",
            {"name": drug_name, "search": "1"},
        )
        rxcui = data.get("idGroup", {}).get("rxnormId", [None])[0]
        if rxcui:
            return Drug(name=drug_name, rxcui=str(rxcui), resolved=True)
        return Drug(name=drug_name, resolved=False)
    except Exception as exc:
        logger.warning("RxNorm lookup failed for %r: %s", drug_name, exc)
        return Drug(name=drug_name, resolved=False)


def _parse_report_sections(report_text: str) -> dict[str, str]:
    """Split the 5-section report into a dict keyed by section name.

    Handles plain ("3. FDA LABEL WARNINGS") and markdown-bold
    ("3. **FDA LABEL WARNINGS**") header formats.
    """
    sections: dict[str, str] = {}
    pattern = re.compile(
        r"\d+\.\s*\**\s*(MEDICATION SUMMARY|INTERACTION ALERTS|FDA LABEL WARNINGS"
        r"|REAL-WORLD ADVERSE EVENTS|RECOMMENDATIONS)\**",
        re.IGNORECASE,
    )
    parts = pattern.split(report_text)
    # parts[0] is the header, then alternating: section_name, content
    for i in range(1, len(parts) - 1, 2):
        name = parts[i].strip().upper()
        content = parts[i + 1].strip()
        # Strip disclaimer/footer that gets captured in the last section's content
        content = re.sub(r"\n---.*", "", content, flags=re.DOTALL).strip()
        sections[name] = content
    return sections


def _rebuild_report(header: str, sections: dict[str, str], footer: str) -> str:
    """Reconstruct the report from a sections dict, preserving order."""
    body_parts = []
    for idx, name in enumerate(SECTION_NAMES, start=1):
        content = sections.get(name, "Data unavailable.")
        body_parts.append(f"{idx}. {name}\n{content}")
    return header + "\n\n" + "\n\n".join(body_parts) + "\n\n" + footer


def _otel_span_set(attributes: dict[str, Any]) -> None:
    """Attach key-value attributes to the current OTel span (best-effort)."""
    try:
        span = trace.get_current_span()
        for k, v in attributes.items():
            span.set_attribute(k, v)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Section-scoped source data filter
# ---------------------------------------------------------------------------


def _source_for_section(section_name: str, source: dict) -> dict:
    """Return only the source data relevant to the given section.

    Prevents the improve LLM from pulling content from other sections
    (e.g. FAERS data leaking into FDA LABEL WARNINGS).
    """
    labels_warnings_only = [
        {
            "drug_name": l["drug_name"],
            "boxed_warning": l.get("boxed_warning", []),
            "warnings_and_cautions": l.get("warnings_and_cautions", []),
            "contraindications": l.get("contraindications", []),
        }
        for l in source.get("labels", [])
    ]
    labels_interactions_only = [
        {
            "drug_name": l["drug_name"],
            "drug_interactions": l.get("drug_interactions", []),
        }
        for l in source.get("labels", [])
    ]

    if section_name == "MEDICATION SUMMARY":
        return {"drugs": source.get("drugs", [])}
    elif section_name == "INTERACTION ALERTS":
        return {
            "interactions": source.get("interactions", []),
            "labels": labels_interactions_only,
        }
    elif section_name == "FDA LABEL WARNINGS":
        return {"labels": labels_warnings_only}
    elif section_name == "REAL-WORLD ADVERSE EVENTS":
        return {"adverse_events": source.get("adverse_events", [])}
    else:  # RECOMMENDATIONS
        return {
            "interactions": source.get("interactions", []),
            "drugs": source.get("drugs", []),
        }


# ---------------------------------------------------------------------------
# Tool 1: resolve_all_medications
# ---------------------------------------------------------------------------


async def resolve_all_medications(medication_text: str) -> str:
    """Parse a natural-language medication list and resolve each drug to an RxCUI code.

    Args:
        medication_text: Free-text description of the patient's medications,
            e.g. "I take metformin, lisinopril, and ibuprofen".

    Returns:
        JSON string: list of {name, rxcui, resolved} objects.
        Unresolvable drugs are included with resolved=false so the report flags them.
    """
    # If the UI already knows the exact drug names (from multiselect), skip Gemini extraction —
    # avoids an unnecessary LLM call and prevents name drift that could cause cache misses.
    if "selected_drugs" in _PIPELINE_CACHE:
        drug_names: list[str] = _PIPELINE_CACHE["selected_drugs"]
    else:
        extraction_prompt = (
            "Extract every substance from the following text that could have a drug interaction. "
            "This includes prescription drugs, OTC medications, supplements, vitamins, herbal products, "
            "and substances like alcohol, caffeine, or nicotine. "
            "Return a JSON array of strings, one name per substance, using its common name only (no dosage, no form). "
            "Output ONLY the JSON array, no explanation.\n\n"
            f"Text: {medication_text}"
        )
        raw = await _generate(extraction_prompt, thinking=False)
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            drug_names = json.loads(clean)
        except json.JSONDecodeError:
            drug_names = [
                d.strip()
                for d in re.split(r"[,;\n]|\band\b", medication_text, flags=re.IGNORECASE)
                if d.strip()
            ]

    # Resolve all drugs in parallel via RxNorm, skipping any already in the pre-resolved cache
    valid_names = [n for n in drug_names if n]
    pre_resolved: dict[str, dict] = _PIPELINE_CACHE.get("pre_resolved", {})

    def _cache_lookup(name: str) -> dict | None:
        key = name.lower()
        if key in pre_resolved:
            return pre_resolved[key]
        # fuzzy: "metformin" matches "metformin hydrochloride" in cache
        for cache_key, val in pre_resolved.items():
            if cache_key.startswith(key) or key.startswith(cache_key):
                return val
        return None

    async def _resolve_or_use_cache(name: str) -> Drug:
        cached = _cache_lookup(name)
        if cached:
            return Drug(**cached)
        return await _resolve_single_rxcui(name)

    drugs: list[Drug] = list(
        await asyncio.gather(*[_resolve_or_use_cache(name) for name in valid_names])
    )

    result = [d.model_dump() for d in drugs]
    _otel_span_set(
        {
            "drugs.requested": len(drug_names),
            "drugs.resolved": sum(1 for d in drugs if d.resolved),
            "drugs.unresolved": sum(1 for d in drugs if not d.resolved),
        }
    )
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Pharmacological knowledge fallback for interaction checking
# ---------------------------------------------------------------------------

_INTERACTION_FALLBACK_PROMPT = """You are a clinical pharmacist. For each drug pair listed below, evaluate whether a clinically relevant interaction exists.

Pairs to evaluate:
{pair_list}

You MUST include one entry per pair, even if the interaction is minor or theoretical.

Severity calibration:
- major: life-threatening or requires discontinuation (e.g. warfarin + aspirin bleeding, serotonin syndrome, QT prolongation)
- moderate: requires monitoring or possible dose adjustment (e.g. ACE inhibitor + antidiabetic hypoglycemia risk, NSAID + ACE inhibitor renal risk)
- minor: any theoretical, pharmacokinetic, or weakly documented interaction worth noting (e.g. renal competition, protein-binding displacement, additive GI effects)
- unknown: no known pharmacological basis found in literature

Return ONLY a JSON array. Each element has exactly these keys:
  "drug1": string, "drug2": string,
  "severity": "major" | "moderate" | "minor" | "unknown",
  "description": ONE sentence max (≤120 chars) — effect + mechanism only,
  "source": "Pharmacological literature"

Output ONLY the JSON array, no markdown."""


async def _gemini_interaction_fallback(pairs: list[tuple[dict, dict]]) -> list[Interaction]:
    """Use Gemini parametric knowledge for drug pairs not covered by NIH."""
    pair_list = "\n".join(f"- {d1['name']} + {d2['name']}" for d1, d2 in pairs)
    prompt = _INTERACTION_FALLBACK_PROMPT.format(pair_list=pair_list)
    try:
        raw = await _generate(prompt, thinking=False)
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        items: list[dict] = json.loads(clean)
        return [
            Interaction(
                drug1_name=item.get("drug1", ""),
                drug2_name=item.get("drug2", ""),
                severity=item.get("severity", "unknown").lower(),
                description=item.get("description", ""),
                source="Pharmacological literature",
            )
            for item in items
            if item.get("drug1") and item.get("drug2")
        ]
    except Exception as exc:
        logger.warning("Gemini interaction fallback failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Tool 2: check_all_interactions
# ---------------------------------------------------------------------------


async def check_all_interactions(resolved_drugs_json: str) -> str:
    """Check NIH Drug Interaction API for every pairwise combination of the resolved drugs.

    Args:
        resolved_drugs_json: JSON string returned by resolve_all_medications.

    Returns:
        JSON string: list of {drug1_name, drug2_name, severity, description, source} objects.
        If no interactions exist between any pair, returns an empty list.
    """
    drugs_raw: list[dict] = json.loads(resolved_drugs_json)
    resolved = [d for d in drugs_raw if d.get("resolved") and d.get("rxcui")]
    _PIPELINE_CACHE["drugs"] = drugs_raw

    async def _check_pair(d1: dict, d2: dict) -> list[Interaction]:
        try:
            url = f"{RXNORM_BASE}/interaction/list.json?rxcuis={d1['rxcui']}+{d2['rxcui']}"
            data = await _http_get_raw_url(url)
            result: list[Interaction] = []
            for group in (data.get("fullInteractionTypeGroup", []) or []):
                for itype in group.get("fullInteractionType", []):
                    for pair in itype.get("interactionPair", []):
                        result.append(Interaction(
                            drug1_name=d1["name"],
                            drug2_name=d2["name"],
                            severity=pair.get("severity", "unknown").lower(),
                            description=pair.get("description", ""),
                            source=group.get("sourceName", "NIH RxNorm"),
                        ))
            return result
        except Exception as exc:
            logger.warning("Interaction check failed for %s/%s: %s", d1["name"], d2["name"], exc)
            return []

    pairs = list(combinations(resolved, 2))
    pair_results = await asyncio.gather(*[_check_pair(d1, d2) for d1, d2 in pairs])

    interactions: list[Interaction] = []
    no_nih_pairs: list[tuple[dict, dict]] = []
    for (d1, d2), result in zip(pairs, pair_results):
        if result:
            interactions.extend(result)
        else:
            no_nih_pairs.append((d1, d2))

    # Unresolved drugs (no rxcui) can't use the NIH API but Gemini can still check
    # interactions for them by name — add all pairs that include an unresolved drug.
    unresolved = [d for d in drugs_raw if not (d.get("resolved") and d.get("rxcui"))]
    for u in unresolved:
        for other in resolved:
            no_nih_pairs.append((u, other))
        for u2 in unresolved:
            if u["name"] < u2["name"]:  # avoid duplicate pairs
                no_nih_pairs.append((u, u2))

    # Per-pair fallback: NIH often has no data for valid interactions.
    # Runs for all pairs not covered by NIH — including unresolved drugs.
    if no_nih_pairs:
        fallback = await _gemini_interaction_fallback(no_nih_pairs)
        existing: set[tuple[str, str]] = {
            (ix.drug1_name.lower(), ix.drug2_name.lower()) for ix in interactions
        } | {(ix.drug2_name.lower(), ix.drug1_name.lower()) for ix in interactions}
        for ix in fallback:
            key = (ix.drug1_name.lower(), ix.drug2_name.lower())
            rev = (key[1], key[0])
            if key not in existing and rev not in existing:
                interactions.append(ix)
                existing.add(key)
                existing.add(rev)

    severity_order = {"major": 0, "moderate": 1, "minor": 2, "unknown": 3}
    interactions.sort(key=lambda x: severity_order.get(x.severity, 3))

    interaction_dicts = [i.model_dump() for i in interactions]
    _PIPELINE_CACHE["interactions"] = interaction_dicts

    _otel_span_set(
        {
            "interactions.total": len(interactions),
            "interactions.major": sum(1 for i in interactions if i.severity == "major"),
            "interactions.moderate": sum(
                1 for i in interactions if i.severity == "moderate"
            ),
            "interactions.minor": sum(1 for i in interactions if i.severity == "minor"),
        }
    )
    return json.dumps(interaction_dicts)


# ---------------------------------------------------------------------------
# FDA label helper — picks the label result richest in drug_interactions data
# ---------------------------------------------------------------------------


async def _fetch_best_label(name: str, rxcui: str) -> DrugLabel:
    """Try up to two search strategies, return the label result with the most data."""
    label_obj = DrugLabel(rxcui=rxcui, drug_name=name)

    searches = [
        f"openfda.generic_name:{name}",
        f"openfda.substance_name:{name}",
    ]

    for search_query in searches:
        data = await _http_get(
            f"{OPENFDA_BASE}/label.json",
            {"search": search_query, "limit": "5"},
        )
        results = data.get("results", [])
        if not results:
            continue

        # Pick the result with the richest drug_interactions field
        best = max(
            results,
            key=lambda r: len(" ".join(r.get("drug_interactions", []) or [])),
        )
        r = best

        label_obj.boxed_warning = [
            _safe_text(w) for w in (r.get("boxed_warning", []) or [])
        ]
        # OTC labels use "warnings" instead of "warnings_and_cautions" — fall back to it
        warnings_raw = r.get("warnings_and_cautions") or r.get("warnings") or []
        label_obj.warnings_and_cautions = [_safe_text(w) for w in warnings_raw]
        label_obj.contraindications = [
            _safe_text(c) for c in (r.get("contraindications", []) or [])
        ]
        label_obj.drug_interactions = [
            _safe_text(i) for i in (r.get("drug_interactions", []) or [])
        ]

        # If we got drug_interactions, no need to try the next strategy
        if label_obj.drug_interactions:
            break

    return label_obj


# ---------------------------------------------------------------------------
# Tool 3: get_fda_enrichment
# ---------------------------------------------------------------------------


async def get_fda_enrichment(resolved_drugs_json: str) -> str:
    """Fetch FDA label warnings and FAERS adverse event reports for each resolved drug.

    Args:
        resolved_drugs_json: JSON string returned by resolve_all_medications.

    Returns:
        JSON string: {"labels": [...], "adverse_events": [...]}.
    """
    drugs_raw: list[dict] = json.loads(resolved_drugs_json)

    async def _fetch_drug_fda(drug: dict) -> tuple[DrugLabel, AdverseEvent]:
        name = drug["name"]
        rxcui = drug.get("rxcui", "")
        label_obj = DrugLabel(rxcui=rxcui, drug_name=name)
        event_obj = AdverseEvent(drug_name=name)

        async def _get_label() -> None:
            nonlocal label_obj
            try:
                label_obj = await _fetch_best_label(name, rxcui)
            except Exception as exc:
                logger.warning("FDA label fetch failed for %s: %s", name, exc)

        async def _get_events() -> None:
            # Try indexed openfda field first — more reliable for high-volume drugs.
            # Fall back to raw medicinalproduct field if the indexed field returns nothing.
            queries = [
                f'patient.drug.openfda.generic_name:"{name}"',
                f'patient.drug.openfda.substance_name:"{name}"',
                f'patient.drug.medicinalproduct:"{name.upper()}"',
            ]
            for query in queries:
                try:
                    data = await _http_get(
                        f"{OPENFDA_BASE}/event.json",
                        {"search": query, "limit": "10"},
                    )
                    if not data.get("results"):
                        continue
                    meta = data.get("meta", {})
                    results = data.get("results", [])
                    event_obj.total_reports = meta.get("results", {}).get("total", 0)
                    event_obj.serious_count = sum(
                        1 for r in results if r.get("serious") in ("1", 1)
                    )
                    seen: list[str] = []
                    for r in results:
                        for rxn in r.get("patient", {}).get("reaction", []):
                            term = _safe_text(rxn.get("reactionmeddrapt", ""))
                            if term and term not in seen:
                                seen.append(term)
                    event_obj.top_reactions = seen
                    return
                except Exception as exc:
                    logger.debug("FAERS query failed for %s (query=%r): %s", name, query, exc)
            logger.warning("FAERS: no data found for %s", name)

        # Label and FAERS fetches for the same drug are independent — run in parallel
        await asyncio.gather(_get_label(), _get_events())
        return label_obj, event_obj

    # All drugs are independent — fetch in parallel
    drug_results = await asyncio.gather(*[_fetch_drug_fda(drug) for drug in drugs_raw])
    labels: list[DrugLabel] = [r[0] for r in drug_results]
    events: list[AdverseEvent] = [r[1] for r in drug_results]

    fda_full = {
        "labels": [l.model_dump() for l in labels],
        "adverse_events": [e.model_dump() for e in events],
    }
    _PIPELINE_CACHE["fda"] = fda_full

    labels_fetched = sum(
        1 for l in labels
        if l.boxed_warning or l.warnings_and_cautions or l.drug_interactions
    )
    events_fetched = sum(1 for e in events if e.total_reports > 0)

    _otel_span_set(
        {
            "fda.labels_fetched": labels_fetched,
            "fda.events_fetched": events_fetched,
        }
    )
    # Return compact summary only — full data is in _PIPELINE_CACHE["fda"]
    return json.dumps({
        "status": "ok",
        "labels_fetched": labels_fetched,
        "events_fetched": events_fetched,
        "drugs": [l.drug_name for l in labels],
    })


# ---------------------------------------------------------------------------
# Tool 4: synthesize_safety_report
# ---------------------------------------------------------------------------


async def synthesize_safety_report(
    resolved_drugs_json: str,
    interactions_json: str,
) -> str:
    """Use Gemini to synthesize all collected data into a structured 5-section safety report.

    Args:
        resolved_drugs_json: JSON string from resolve_all_medications.
        interactions_json: JSON string from check_all_interactions.
        historical_weaknesses: Optional. Comma-separated section names that have averaged
            below 0.85 in recent runs (from local run_history.json). When provided,
            the synthesis prompt directs extra rigor to those sections.

    Returns:
        Formatted report text (the 5-section structure defined in the system prompt).
    """
    drugs: list[dict] = json.loads(resolved_drugs_json)
    interactions: list[dict] = (
        _PIPELINE_CACHE["interactions"] if "interactions" in _PIPELINE_CACHE
        else json.loads(interactions_json)
    )
    fda: dict = _PIPELINE_CACHE.get("fda", {})

    drugs_summary = "\n".join(
        f"- {d['name']}: RxCUI={d.get('rxcui', 'N/A')}, resolved={d['resolved']}"
        for d in drugs
    )

    if interactions:
        interactions_summary = "\n".join(
            f"- [{i['severity'].upper()}] {i['drug1_name']} + {i['drug2_name']}: "
            f"{i['description']} (source: {i['source']})"
            for i in interactions
        )
    else:
        interactions_summary = "No known interactions found between the listed medications."

    labels_summary = ""
    for label in fda.get("labels", []):
        parts = [f"Drug: {label['drug_name']}"]
        if label.get("boxed_warning"):
            parts.append("Boxed Warning: " + " | ".join(str(w) for w in label["boxed_warning"]))
        if label.get("warnings_and_cautions"):
            parts.append("Warnings: " + " | ".join(str(w) for w in label["warnings_and_cautions"]))
        if label.get("contraindications"):
            parts.append("Contraindications: " + " | ".join(str(c) for c in label["contraindications"]))
        if label.get("drug_interactions"):
            parts.append("Drug Interactions (per FDA label): " + " | ".join(str(i) for i in label["drug_interactions"]))
        if len(parts) > 1:
            labels_summary += "\n".join(parts) + "\n\n"
    if not labels_summary:
        labels_summary = "FDA label data unavailable for all listed drugs."

    events_summary = ""
    for ev in fda.get("adverse_events", []):
        if ev.get("total_reports", 0) > 0:
            events_summary += (
                f"Drug: {ev['drug_name']} — {ev['total_reports']:,} total FAERS reports, "
                f"{ev['serious_count']} serious in sample. "
                f"Top reactions: {', '.join(ev['top_reactions'])}\n"
            )
    if not events_summary:
        events_summary = "Adverse event data unavailable."

    _hw = _PIPELINE_CACHE.get("historical_weaknesses", "").strip()
    historical_focus = (
        f"\nHISTORICAL QUALITY SIGNAL — The judge has flagged these sections as "
        f"below target in past runs, with specific failure reasons recorded:\n{_hw}\n"
        f"You MUST avoid repeating these exact mistakes. Address each logged failure "
        f"reason directly when writing the relevant section.\n"
        if _hw
        else ""
    )
    prompt = SYNTHESIS_PROMPT.format(
        drugs_summary=drugs_summary,
        interactions_summary=interactions_summary,
        labels_summary=labels_summary,
        events_summary=events_summary,
        historical_focus=historical_focus,
    )

    report = await _generate(prompt, thinking=False)
    _PIPELINE_CACHE["report"] = report
    _otel_span_set({"report.char_count": len(report)})
    return report


# ---------------------------------------------------------------------------
# Tool 5: evaluate_report_quality
# ---------------------------------------------------------------------------


async def evaluate_report_quality(report_text: str) -> str:
    """Run LLM-as-a-Judge evaluation on each report section, log scores as OTel span attributes.

    Scores each section on accuracy, caution, clarity, and citation (each 0.0–1.0).
    The overall section score is the mean. Sections scoring below 0.55 are rewritten by
    apply_self_improvement. Judge criteria are fetched from Phoenix Prompt Management at
    runtime so they can be updated without code changes; falls back to JUDGE_BATCH_PROMPT.

    Returns:
        JSON string: list of {section, accuracy, caution, clarity, citation, overall, reasoning}.
    """
    fda = _PIPELINE_CACHE.get("fda", {})
    interactions_cached = _PIPELINE_CACHE.get("interactions", [])

    source_summary = json.dumps(
        {
            "drugs": _PIPELINE_CACHE.get("drugs", []),
            "interactions": interactions_cached,
            "labels": [
                {
                    "drug": l["drug_name"],
                    "boxed_warning": l.get("boxed_warning", []),
                    "drug_interactions": l.get("drug_interactions", []),
                }
                for l in fda.get("labels", [])
            ],
            "adverse_events": [
                {
                    "drug": e["drug_name"],
                    "total_reports": e.get("total_reports", 0),
                    "serious_count": e.get("serious_count", 0),
                    "reactions": e["top_reactions"],
                }
                for e in fda.get("adverse_events", [])
            ],
        },
        indent=None,
    )

    # Fetch judge criteria from Phoenix Prompt Management via Python SDK.
    # Falls back to the built-in JUDGE_BATCH_PROMPT if Phoenix is unreachable.
    # str.replace instead of .format(): report_text may contain {curly braces} from
    # pharmacological notation; Phoenix Mustache templates use {{placeholder}} syntax.
    from phoenix_client import get_judge_criteria, log_eval_annotations
    _phoenix_criteria, _prompt_version_id = await get_judge_criteria()
    _criteria = _phoenix_criteria
    _PIPELINE_CACHE["judge_prompt_version"] = _prompt_version_id or "built-in"

    if _criteria:
        # Phoenix prompt uses Mustache {{placeholder}} syntax
        batch_prompt = (
            _criteria
            .replace("{{report_text}}", report_text)
            .replace("{{source_data}}", source_summary)
        )
    else:
        batch_prompt = (
            JUDGE_BATCH_PROMPT
            .replace("{report_text}", report_text)
            .replace("{source_data}", source_summary)
        )

    eval_scores: list[SectionEval] = []
    raw_batch: dict = {}

    try:
        raw = await _generate(batch_prompt, thinking=False)
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            raw_batch = json.loads(clean)
        except json.JSONDecodeError:
            # Extract outermost {...} in case LLM added surrounding text
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if m:
                candidate = m.group()
                # Fix trailing commas before } or ]
                candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                raw_batch = json.loads(candidate)
            else:
                raise
    except Exception as exc:
        logger.warning("Batch judge eval failed: %s", exc)

    for section_name in SECTION_NAMES:
        scores = raw_batch.get(section_name, {})
        if scores:
            ev = SectionEval(
                section=section_name,
                accuracy=float(scores.get("accuracy", 0.5)),
                caution=float(scores.get("caution", 0.5)),
                clarity=float(scores.get("clarity", 0.5)),
                citation=float(scores.get("citation", 0.5)),
                reasoning=scores.get("reasoning", ""),
            )
        else:
            ev = SectionEval(
                section=section_name,
                accuracy=0.5,
                caution=0.5,
                clarity=0.5,
                citation=0.5,
                reasoning="Section not found or evaluation failed.",
            )

        eval_scores.append(ev)

        # Log each section's scores as OTel span attributes so they appear in Phoenix
        safe_name = section_name.lower().replace(" ", "_").replace("&", "and")
        _otel_span_set(
            {
                f"eval.{safe_name}.accuracy": ev.accuracy,
                f"eval.{safe_name}.caution": ev.caution,
                f"eval.{safe_name}.clarity": ev.clarity,
                f"eval.{safe_name}.citation": ev.citation,
                f"eval.{safe_name}.overall": ev.overall,
            }
        )

    below_threshold = [e for e in eval_scores if e.overall < 0.55]
    _otel_span_set(
        {
            "eval.sections_below_threshold": len(below_threshold),
            "eval.sections_needing_improvement": ",".join(
                e.section for e in below_threshold
            ),
        }
    )

    eval_dicts = [e.model_dump() for e in eval_scores]
    _PIPELINE_CACHE["eval_scores"] = eval_dicts

    # Log scores as span annotations — visible as first-class quality scores in Phoenix UI.
    # Also record which prompt version was used so every trace is auditable.
    span_ctx = trace.get_current_span().get_span_context()
    if span_ctx.is_valid:
        _otel_span_set({"eval.judge_prompt_version": _PIPELINE_CACHE.get("judge_prompt_version", "built-in")})
        log_eval_annotations(format(span_ctx.span_id, "016x"), eval_dicts)

    return json.dumps(eval_dicts)


# ---------------------------------------------------------------------------
# Tool 6: apply_self_improvement
# ---------------------------------------------------------------------------


async def apply_self_improvement(report_text: str) -> str:
    """Rerun low-confidence report sections with conservative, citation-heavy prompting.

    Identifies sections with overall confidence < 0.7 from the pipeline cache
    (populated by evaluate_report_quality) and regenerates those sections with
    stricter prompting. Logs before/after score deltas as OTel span attributes
    visible in Phoenix traces.

    Returns:
        The final improved report text with updated sections and a per-section
        confidence summary embedded in section 5.
    """
    eval_scores: list[dict] = _PIPELINE_CACHE.get("eval_scores", [])
    fda = _PIPELINE_CACHE.get("fda", {})
    source = {
        "drugs": _PIPELINE_CACHE.get("drugs", []),
        "interactions": _PIPELINE_CACHE.get("interactions", []),
        "labels": fda.get("labels", []),
        "adverse_events": fda.get("adverse_events", []),
    }
    report_text = _PIPELINE_CACHE.get("report", report_text)
    sections = _parse_report_sections(report_text)

    # Guard: if parsing failed, return the original report unchanged
    if len(sections) < 3:
        logger.warning("Section parsing failed (%d sections found); appending scores only.", len(sections))
        score_lines = ["Per-section confidence scores (0.0–1.0):"]
        for ev in eval_scores:
            overall = ev.get("overall")
            if overall is None:
                overall = sum(ev.get(d, 0.5) for d in ["accuracy", "caution", "clarity", "citation"]) / 4.0
            score_lines.append(f"  • {ev['section']}: {overall:.2f}")
        score_block = "\n".join(score_lines)
        # Insert scores before the disclaimer line, not after it
        disclaimer_match = re.search(r"\n---\s*\nDISCLAIMER", report_text, re.IGNORECASE)
        if disclaimer_match:
            insert_at = disclaimer_match.start()
            final_report = report_text[:insert_at] + "\n\n" + score_block + report_text[insert_at:]
        else:
            final_report = report_text.rstrip() + "\n\n" + score_block + "\n"
        _PIPELINE_CACHE["report"] = final_report
        return final_report

    # Extract header and footer from the original report
    header_match = re.match(r"(.*?)\n\s*1\.", report_text, re.DOTALL)
    header = header_match.group(1).strip() if header_match else "=== DRUG INTERACTION SAFETY REPORT ==="
    footer_match = re.search(r"(---.*?DISCLAIMER.*)", report_text, re.DOTALL | re.IGNORECASE)
    footer = (
        footer_match.group(1).strip()
        if footer_match
        else (
            "---\n"
            "DISCLAIMER: This report is for informational purposes only and is not "
            "medical advice. Always discuss medication interactions with your doctor "
            "or pharmacist before making any changes to your medications."
        )
    )

    source_summary = json.dumps(
        {
            "interactions": source["interactions"],
            "labels": [
                {k: v for k, v in l.items() if k in ("drug_name", "boxed_warning", "drug_interactions")}
                for l in source["labels"]
            ],
            "adverse_events": [
                {
                    "drug_name": e["drug_name"],
                    "total_reports": e.get("total_reports", 0),
                    "serious_count": e.get("serious_count", 0),
                    "top_reactions": e.get("top_reactions", []),
                }
                for e in source["adverse_events"]
            ],
        }
    )

    improved_sections = dict(sections)
    improvement_log: list[dict] = []

    async def _improve_section(eval_entry: dict) -> dict | None:
        section_name = eval_entry["section"].upper()
        if "overall" in eval_entry:
            before_score = eval_entry["overall"]
        else:
            dims = ["accuracy", "caution", "clarity", "citation"]
            before_score = sum(eval_entry.get(d, 0.5) for d in dims) / 4.0

        if before_score >= 0.55:
            return None

        original_content = sections.get(section_name, "")
        if not original_content:
            return None

        section_source = _source_for_section(section_name, source)
        improve_prompt = IMPROVE_PROMPT.format(
            section_name=section_name,
            score=before_score,
            reasoning=eval_entry.get("reasoning", "Quality below threshold."),
            original_content=original_content,
            source_data=json.dumps(section_source),
        )

        try:
            improved_content = await _generate(improve_prompt, thinking=False)
            # Estimate after_score rather than making a second Gemini call — avoids
            # back-to-back quota hits while still showing a meaningful delta in the UI.
            after_score = min(before_score + 0.15, 1.0)
            return {"section_name": section_name, "content": improved_content.strip(),
                    "before": before_score, "after": after_score}

        except Exception as exc:
            logger.warning("Improvement failed for %r: %s", section_name, exc)
            return {"section_name": section_name, "content": original_content,
                    "before": before_score, "after": before_score}

    improve_results = [await _improve_section(ev) for ev in eval_scores]

    # Collect all rewritten sections, then re-score them in one batch call
    rewritten = [r for r in improve_results if r is not None]
    for r in rewritten:
        improved_sections[r["section_name"]] = r["content"]

    # One batch re-evaluation for all improved sections — real scores, 1 Gemini call
    if rewritten:
        partial_report = _rebuild_report(header, improved_sections, footer)
        try:
            batch_prompt = (
                JUDGE_BATCH_PROMPT
                .replace("{report_text}", partial_report)
                .replace("{source_data}", source_summary)
            )
            raw = await _generate(batch_prompt, thinking=False)
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            try:
                reeval_batch: dict = json.loads(clean)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", clean, re.DOTALL)
                if m:
                    candidate = re.sub(r",\s*([}\]])", r"\1", m.group())
                    reeval_batch = json.loads(candidate)
                else:
                    raise
        except Exception as exc:
            logger.warning("Batch re-evaluation after improvement failed: %s", exc)
            reeval_batch = {}

        for r in rewritten:
            section_name = r["section_name"]
            scores = reeval_batch.get(section_name, {})
            dims = ["accuracy", "caution", "clarity", "citation"]
            if scores:
                after_score = sum(float(scores.get(d, 0.5)) for d in dims) / 4.0
            else:
                after_score = r["before"]
            before_score = r["before"]
            # Revert if the rewrite made things worse — keep original content
            if after_score < before_score:
                improved_sections[section_name] = sections.get(section_name, r["content"])
                after_score = before_score
            improvement_log.append({
                "section": section_name,
                "before": round(before_score, 3),
                "after": round(after_score, 3),
                "delta": round(after_score - before_score, 3),
            })
            safe_name = section_name.lower().replace(" ", "_").replace("&", "and")
            _otel_span_set({
                f"improvement.{safe_name}.before": round(before_score, 3),
                f"improvement.{safe_name}.after": round(after_score, 3),
                f"improvement.{safe_name}.delta": round(after_score - before_score, 3),
            })

    # Rebuild section 5 with actual confidence scores table
    score_table_lines = ["Per-section confidence scores (0.0–1.0):"]
    for ev in eval_scores:
        section = ev["section"]
        overall = ev.get("overall")
        if overall is None:
            dims = ["accuracy", "caution", "clarity", "citation"]
            overall = sum(ev.get(d, 0.5) for d in dims) / 4.0
        improved_marker = ""
        for log_entry in improvement_log:
            if log_entry["section"] == section.upper():
                improved_marker = f" → improved to {log_entry['after']:.2f}"
                break
        score_table_lines.append(f"  • {section}: {overall:.2f}{improved_marker}")

    existing_confidence = improved_sections.get("RECOMMENDATIONS", "")
    # Strip any existing score table if present, then append fresh one
    existing_confidence = re.sub(
        r"Per-section confidence scores.*", "", existing_confidence, flags=re.DOTALL
    ).strip()
    improved_sections["RECOMMENDATIONS"] = (
        existing_confidence + "\n\n" + "\n".join(score_table_lines)
    )

    final_report = _rebuild_report(header, improved_sections, footer)
    _PIPELINE_CACHE["improvement_log"] = improvement_log

    total_improvements = len(improvement_log)
    avg_delta = (
        sum(e["delta"] for e in improvement_log) / total_improvements
        if total_improvements
        else 0.0
    )
    _otel_span_set(
        {
            "self_improve.sections_rerun": total_improvements,
            "self_improve.avg_score_delta": round(avg_delta, 3),
        }
    )

    _PIPELINE_CACHE["report"] = final_report

    # Merge after-improvement scores back into eval_scores so Phoenix dataset and
    # span annotations record the FINAL quality, not the pre-improvement baseline.
    after_score_map = {log["section"]: log["after"] for log in improvement_log}
    final_eval_scores = []
    for ev in eval_scores:
        section = ev["section"].upper()
        if section in after_score_map:
            ev = dict(ev)
            ev["overall"] = after_score_map[section]
        final_eval_scores.append(ev)
    _PIPELINE_CACHE["eval_scores"] = final_eval_scores

    from phoenix_client import log_eval_annotations, write_run_scores
    await write_run_scores(
        drugs=_PIPELINE_CACHE.get("drugs", []),
        eval_scores=final_eval_scores,
        sections_improved=[log["section"] for log in improvement_log],
    )
    _PIPELINE_CACHE["historical_runs"] = _PIPELINE_CACHE.get("historical_runs", 0) + 1

    # Re-log span annotations with final scores so Phoenix traces show post-improvement values.
    span_ctx = trace.get_current_span().get_span_context()
    if span_ctx.is_valid and improvement_log:
        log_eval_annotations(format(span_ctx.span_id, "016x"), final_eval_scores)

    return final_report
