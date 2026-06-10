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

"""Phoenix integration — MCP for dataset/prompt operations, Python SDK for tracing.

Dataset reads/writes and prompt fetches go through the Phoenix MCP server
(@arizeai/phoenix-mcp via npx) so they appear as first-class tool calls in
traces. Span annotations and prompt syncing use the Python SDK directly.
All operations are best-effort: failures fall back gracefully so a Phoenix
outage never breaks the pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

_DATASET_NAME = "drug-interaction-eval-history"
_PROMPT_IDENTIFIER = "druginteractionqualitycriteria"
_SECTION_NAMES = [
    "MEDICATION SUMMARY",
    "INTERACTION ALERTS",
    "FDA LABEL WARNINGS",
    "REAL-WORLD ADVERSE EVENTS",
    "RECOMMENDATIONS",
]

# Active MCP call function for the current pipeline run — set by phoenix_mcp_session().
_active_mcp_call = None


# ---------------------------------------------------------------------------
# MCP session management
# ---------------------------------------------------------------------------


class phoenix_mcp_session:
    """Async context manager that opens one Phoenix MCP session per pipeline run.

    Sets module-level _active_mcp_call so all Phoenix functions use MCP
    automatically. Falls back silently if env vars are missing or npx fails,
    so the pipeline continues with Python SDK fallbacks.

    Uses a class-based implementation (not @asynccontextmanager) to avoid
    generator-based async context manager edge cases ("generator didn't stop
    after athrow()") that occur when pipeline exceptions propagate through
    generator cleanup.
    """

    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._session = None

    async def __aenter__(self) -> "phoenix_mcp_session":
        global _active_mcp_call

        base_url = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
        api_key = os.environ.get("PHOENIX_API_KEY", "").strip()
        if not base_url or not api_key:
            return self

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command="npx",
                args=["@arizeai/phoenix-mcp", "--baseUrl", base_url, "--apiKey", api_key],
                env={
                    **dict(os.environ),
                    "PHOENIX_API_KEY": api_key,
                    "PHOENIX_BASE_URL": base_url,
                    "PHOENIX_COLLECTOR_ENDPOINT": base_url,
                },
            )
            self._stack = AsyncExitStack()
            read, write = await self._stack.enter_async_context(stdio_client(server_params))
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()

            session = self._session

            async def _call(tool_name: str, arguments: dict):
                try:
                    result = await session.call_tool(tool_name, arguments)
                    for item in result.content:
                        text = getattr(item, "text", None)
                        if not text or not text.strip():
                            continue
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            logger.warning("MCP tool %r non-JSON response: %r", tool_name, text[:120])
                except Exception as exc:
                    logger.warning("MCP tool %r failed: %s", tool_name, exc)
                return None

            _active_mcp_call = _call

        except Exception as exc:
            logger.warning("Phoenix MCP session failed to start: %s", exc)
            if self._stack is not None:
                try:
                    await self._stack.aclose()
                except Exception:
                    pass
                self._stack = None

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        global _active_mcp_call
        _active_mcp_call = None
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception as exc:
                logger.warning("Phoenix MCP session cleanup failed: %s", exc)
            self._stack = None
        return False  # never suppress exceptions from the pipeline body


# ---------------------------------------------------------------------------
# Python SDK client (tracing, prompt sync, span annotations)
# ---------------------------------------------------------------------------


def _get_client():
    """Return a configured Phoenix REST client, or None if env vars are missing."""
    base_url = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
    api_key = os.environ.get("PHOENIX_API_KEY", "").strip()
    if not base_url or not api_key:
        return None
    try:
        from phoenix.client import Client
        return Client(base_url=base_url, api_key=api_key)
    except Exception as exc:
        logger.warning("Phoenix client unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Prompt Management
# ---------------------------------------------------------------------------


def _extract_prompt_text(pv) -> str:
    """Extract template text from a Python SDK PromptVersion object."""
    for msg in pv._template.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part["text"]
    return ""


def _extract_mcp_prompt_text(data: dict) -> str:
    """Extract template text from a Phoenix MCP get-prompt-by-identifier response.

    The MCP response has template as a nested dict:
    {"template": {"type": "chat", "messages": [{"role": "user", "content": "..."}]}}
    """
    tmpl = data.get("template", {})
    if isinstance(tmpl, str):
        return tmpl
    if isinstance(tmpl, dict):
        for msg in tmpl.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                return content
    return ""


async def get_judge_criteria() -> tuple[str, str]:
    """Fetch the versioned judge criteria from Phoenix via MCP.

    Returns (template_text, version_id). Falls back to Python SDK, then
    empty strings if both fail (pipeline uses built-in JUDGE_BATCH_PROMPT).
    """
    if _active_mcp_call:
        data = await _active_mcp_call(
            "get-prompt-by-identifier",
            {"prompt_identifier": _PROMPT_IDENTIFIER},
        )
        if data:
            text = _extract_mcp_prompt_text(data)
            version_id = data.get("id", "") or data.get("version_id", "")
            if text:
                return text, version_id

    # Python SDK fallback
    client = _get_client()
    if not client:
        return "", ""
    try:
        pv = client.prompts.get(prompt_identifier=_PROMPT_IDENTIFIER)
        text = _extract_prompt_text(pv)
        return text, (pv.id or "")
    except Exception as exc:
        logger.warning("Could not fetch judge criteria: %s", exc)
    return "", ""


def _code_prompt_as_mustache() -> str:
    """Return JUDGE_BATCH_PROMPT converted to Mustache syntax for Phoenix storage."""
    _agent_dir = str(Path(__file__).resolve().parent)
    if _agent_dir not in sys.path:
        sys.path.insert(0, _agent_dir)
    from drug_interaction_agent.prompts import JUDGE_BATCH_PROMPT
    return (
        JUDGE_BATCH_PROMPT
        .replace("{report_text}", "{{report_text}}")
        .replace("{source_data}", "{{source_data}}")
    )


def _push_prompt_version(client, template_text: str) -> None:
    """Push a new prompt version to Phoenix Prompt Management."""
    from phoenix.client.types.prompts import PromptVersion
    from phoenix.client.__generated__ import v1

    message = v1.PromptMessage(role="user", content=template_text)
    prompt_version = PromptVersion(
        [message],
        model_name=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        model_provider="GOOGLE",
        template_format="MUSTACHE",
        description="Evaluation criteria for scoring drug interaction safety report sections on accuracy, caution, clarity, and citation.",
    )
    client.prompts.create(
        version=prompt_version,
        name=_PROMPT_IDENTIFIER,
        prompt_description=(
            "LLM-as-a-Judge criteria for the Drug Interaction Safety Report pipeline. "
            "Scores each of the 5 report sections on accuracy, caution, clarity, and "
            "citation (0.0–1.0 each). Auto-synced from JUDGE_BATCH_PROMPT in code."
        ),
    )


async def ensure_prompt_synced() -> None:
    """Sync JUDGE_BATCH_PROMPT to Phoenix via MCP upsert-prompt, or Python SDK fallback.

    Reads the current Phoenix prompt first. Only pushes a new version when the
    local template differs, so repeated pipeline runs don't create duplicate versions.
    """
    local_text = _code_prompt_as_mustache()

    if _active_mcp_call:
        existing = await _active_mcp_call(
            "get-prompt-by-identifier",
            {"prompt_identifier": _PROMPT_IDENTIFIER},
        )
        current_text = _extract_mcp_prompt_text(existing) if existing else ""

        if current_text.strip() == local_text.strip():
            return

        result = await _active_mcp_call(
            "upsert-prompt",
            {
                "name": _PROMPT_IDENTIFIER,
                "description": (
                    "LLM-as-a-Judge criteria for the Drug Interaction Safety Report pipeline. "
                    "Scores each of the 5 report sections on accuracy, caution, clarity, and "
                    "citation (0.0–1.0 each). Auto-synced from JUDGE_BATCH_PROMPT in code."
                ),
                "template": local_text,
                "model_provider": "GOOGLE",
                "model_name": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            },
        )
        if result is not None:
            logger.info("Judge criteria prompt synced to Phoenix via MCP upsert-prompt.")
            return

    # Python SDK fallback
    client = _get_client()
    if not client:
        return
    try:
        try:
            pv = client.prompts.get(prompt_identifier=_PROMPT_IDENTIFIER)
            phoenix_text = _extract_prompt_text(pv)
            if phoenix_text.strip() == local_text.strip():
                return
            _push_prompt_version(client, local_text)
            logger.info("Judge criteria prompt updated in Phoenix.")
        except ValueError:
            _push_prompt_version(client, local_text)
            logger.info("Judge criteria prompt created in Phoenix Prompt Management.")
    except Exception as exc:
        logger.warning("Could not sync judge criteria prompt to Phoenix: %s", exc)


# ---------------------------------------------------------------------------
# Dataset — historical eval scores
# ---------------------------------------------------------------------------


def _parse_examples_output(examples: list) -> tuple[dict[str, list[float]], dict[str, list[str]]]:
    """Parse a list of dataset examples into score totals and reasoning lists."""
    totals: dict[str, list[float]] = {}
    reasons: dict[str, list[str]] = {}
    for ex in examples:
        if isinstance(ex, dict):
            output: dict = ex.get("output", {})
        else:
            output = getattr(ex, "output", None) or {}
            if not isinstance(output, dict):
                output = {}
        for section in _SECTION_NAMES:
            entry = output.get(section)
            if entry is None:
                continue
            if isinstance(entry, dict):
                score = entry.get("score")
                reasoning = entry.get("reasoning", "").strip()
            else:
                score = entry
                reasoning = ""
            if score is not None:
                totals.setdefault(section, []).append(float(score))
                if reasoning:
                    reasons.setdefault(section, []).append(reasoning)
    return totals, reasons


def _build_weakness_strings(weak: list[str], reasons: dict[str, list[str]]) -> tuple[str, str]:
    """Build display_text (names only) and prompt_text (with reasons) for weak sections."""
    display_text = ", ".join(weak)
    prompt_parts = []
    for section in weak:
        section_reasons = reasons.get(section, [])
        if section_reasons:
            unique = list(dict.fromkeys(section_reasons))[-3:]
            reasons_text = "; ".join(f'"{r}"' for r in unique)
            prompt_parts.append(f"- {section}: {reasons_text}")
        else:
            prompt_parts.append(f"- {section}: scored below target in past runs")
    return display_text, "\n".join(prompt_parts)


async def get_historical_weaknesses() -> tuple[str, str, int]:
    """Read eval history from Phoenix to identify chronically weak sections.

    Uses MCP get-dataset-examples when a session is active; falls back to
    the Python SDK. Returns (display_text, prompt_text, run_count).
    display_text has section names only (for UI); prompt_text includes
    specific judge failure reasons (injected into the synthesis prompt).
    """
    if _active_mcp_call:
        data = await _active_mcp_call(
            "get-dataset-examples",
            {"dataset_name": _DATASET_NAME},
        )
        if data is not None:
            # MCP response: {"data": {"examples": [...], "dataset_id": ...}}
            if isinstance(data, list):
                examples = data
            elif isinstance(data, dict):
                inner = data.get("data", data)
                if isinstance(inner, dict):
                    examples = inner.get("examples", [])
                elif isinstance(inner, list):
                    examples = inner
                else:
                    examples = []
            else:
                examples = []
            if not examples:
                return "", "", 0
            totals, reasons = _parse_examples_output(examples)
            weak = [s for s, scores in totals.items() if sum(scores) / len(scores) < 0.85]
            if not weak:
                return "", "", len(examples)
            display_text, prompt_text = _build_weakness_strings(weak, reasons)
            return display_text, prompt_text, len(examples)

    # Python SDK fallback
    client = _get_client()
    if not client:
        return "", "", 0
    try:
        dataset = client.datasets.get_dataset(dataset=_DATASET_NAME)
        examples = list(dataset)
        if not examples:
            return "", "", 0
        totals, reasons = _parse_examples_output(examples)
        weak = [s for s, scores in totals.items() if sum(scores) / len(scores) < 0.85]
        if not weak:
            return "", "", len(examples)
        display_text, prompt_text = _build_weakness_strings(weak, reasons)
        return display_text, prompt_text, len(examples)
    except ValueError as exc:
        if "not found" in str(exc).lower():
            return "", "", 0
        logger.warning("Could not read Phoenix dataset: %s", exc)
    except Exception as exc:
        logger.warning("Could not read Phoenix dataset: %s", exc)
    return "", "", 0


async def write_run_scores(
    drugs: list[dict],
    eval_scores: list[dict],
    sections_improved: list[str] | None = None,
) -> None:
    """Write this run's per-section eval scores to Phoenix via MCP.

    Uses MCP add-dataset-examples when a session is active; falls back to
    the Python SDK. Creates the dataset automatically on first run.
    """
    if sections_improved is None:
        sections_improved = []

    drug_names = [d["name"] for d in drugs if d.get("resolved")]
    score_output = {
        e["section"]: {
            "score": round(e.get("overall", 0.5), 3),
            "reasoning": e.get("reasoning", ""),
        }
        for e in eval_scores
    }
    example = {
        "input": {"drugs": drug_names, "drug_count": len(drug_names)},
        "output": score_output,
        "metadata": {"sections_improved": sections_improved, "run_date": str(date.today())},
    }

    if _active_mcp_call:
        result = await _active_mcp_call(
            "add-dataset-examples",
            {"dataset_name": _DATASET_NAME, "examples": [example]},
        )
        if result is not None:
            return  # MCP succeeded

    # Python SDK fallback
    client = _get_client()
    if not client:
        return
    try:
        inputs = [example["input"]]
        outputs = [example["output"]]
        metadata = [example["metadata"]]
        try:
            client.datasets.add_examples_to_dataset(
                dataset=_DATASET_NAME,
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
            )
        except ValueError as exc:
            if "not found" in str(exc).lower():
                client.datasets.create_dataset(
                    name=_DATASET_NAME,
                    dataset_description=(
                        "Per-section LLM-as-a-Judge quality scores for each pipeline run. "
                        "Used for cross-run learning: sections averaging below 0.85 "
                        "receive focused synthesis prompting on the next run."
                    ),
                    inputs=inputs,
                    outputs=outputs,
                    metadata=metadata,
                )
            else:
                raise
    except Exception as exc:
        logger.warning("Could not write to Phoenix dataset: %s", exc)


# ---------------------------------------------------------------------------
# Experiments — benchmark dataset
# ---------------------------------------------------------------------------


_BENCHMARK_DATASET_NAME = "drug-interaction-benchmark"

_BENCHMARK_CASES = [
    {
        "medication_text": "I take warfarin and aspirin",
        "description": "Classic major bleeding interaction — two anticoagulants",
        "expected_severity": "major",
    },
    {
        "medication_text": "I take metformin, lisinopril, and ibuprofen",
        "description": "Two major interactions: ACE inhibitor + NSAID renal risk; NSAID reduces metformin efficacy",
        "expected_severity": "major",
    },
    {
        "medication_text": "I take lisinopril and potassium supplements",
        "description": "Hyperkalemia risk — ACE inhibitor retains potassium",
        "expected_severity": "major",
    },
    {
        "medication_text": "I take sertraline and tramadol",
        "description": "Serotonin syndrome risk — SSRI + opioid with serotonergic activity",
        "expected_severity": "major",
    },
    {
        "medication_text": "I take atorvastatin and amlodipine",
        "description": "Minor interaction — calcium channel blocker slightly raises statin levels",
        "expected_severity": "minor",
    },
    {
        "medication_text": "I take warfarin, aspirin, and clopidogrel",
        "description": "Triple antithrombotic therapy — highest bleeding risk combination",
        "expected_severity": "major",
    },
    {
        "medication_text": "I take lisinopril",
        "description": "Single drug — no interactions expected",
        "expected_severity": "none",
    },
]


def get_or_create_benchmark_dataset():
    """Return the benchmark Dataset object, creating it from BENCHMARK_CASES if it doesn't exist."""
    client = _get_client()
    if not client:
        raise RuntimeError("Phoenix client unavailable — check PHOENIX_API_KEY and PHOENIX_COLLECTOR_ENDPOINT")

    try:
        return client.datasets.get_dataset(dataset=_BENCHMARK_DATASET_NAME)
    except ValueError:
        pass

    inputs = [{"medication_text": c["medication_text"]} for c in _BENCHMARK_CASES]
    metadata = [{"description": c["description"], "expected_severity": c["expected_severity"]} for c in _BENCHMARK_CASES]
    return client.datasets.create_dataset(
        name=_BENCHMARK_DATASET_NAME,
        dataset_description=(
            "Fixed benchmark test cases for the Drug Interaction Monitor pipeline. "
            "Used for systematic quality measurement across pipeline versions. "
            "Run via: make experiment"
        ),
        inputs=inputs,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Span Annotations
# ---------------------------------------------------------------------------


def log_eval_annotations(span_id: str, eval_scores: list[dict]) -> None:
    """Attach eval scores as annotations on the current trace in Phoenix."""
    if not span_id:
        return
    client = _get_client()
    if not client:
        return
    try:
        from phoenix.client.resources.spans import SpanAnnotationData

        annotations = []
        for ev in eval_scores:
            overall = ev.get("overall")
            if overall is None:
                dims = ["accuracy", "caution", "clarity", "citation"]
                overall = sum(ev.get(d, 0.5) for d in dims) / 4.0
            label = "high" if overall >= 0.85 else "medium" if overall >= 0.70 else "low"
            annotations.append(
                SpanAnnotationData(
                    name=ev["section"],
                    span_id=span_id,
                    annotator_kind="LLM",
                    result={
                        "label": label,
                        "score": round(overall, 3),
                        "explanation": ev.get("reasoning", ""),
                    },
                )
            )

        if annotations:
            client.spans.log_span_annotations(span_annotations=annotations)
    except Exception as exc:
        logger.warning("Could not log span annotations to Phoenix: %s", exc)
