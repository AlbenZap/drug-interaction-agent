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

"""Direct pipeline runner — calls all 6 tools in Python, no ADK orchestration.

Compared to the ADK InMemoryRunner, this eliminates 6+ Gemini round-trips that
the agent used to spend deciding which tool to call next. Tools 2 and 3 also
run in parallel since both only depend on the resolved drug list.

Each tool call gets its own OTel child span so Phoenix traces retain the same
hierarchical tool-call view as the ADK-instrumented version.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from opentelemetry import trace as otel_trace
from opentelemetry.trace import StatusCode

from instrumentation import setup_tracing
from drug_interaction_agent.tools import (
    _PIPELINE_CACHE,
    apply_self_improvement,
    check_all_interactions,
    evaluate_report_quality,
    get_fda_enrichment,
    resolve_all_medications,
    synthesize_safety_report,
)
from phoenix_client import ensure_prompt_synced, get_historical_weaknesses, phoenix_mcp_session

logger = logging.getLogger(__name__)

_OBSERVABILITY_HEADER = "PIPELINE OBSERVABILITY"


async def _run_with_span(tracer, span_name: str, coro, input_attrs: dict | None = None):
    """Run a coroutine inside a named child span, recording input/output as attributes."""
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("openinference.span.kind", "TOOL")
        span.set_attribute("tool.name", span_name)
        if input_attrs:
            for k, v in input_attrs.items():
                span.set_attribute(k, str(v))
        result = await coro
        if result is not None:
            out = str(result)
            span.set_attribute("output.value", out)
            stripped = out.lstrip()
            mime = "application/json" if stripped and stripped[0] in ("{", "[") else "text/plain"
            span.set_attribute("output.mime_type", mime)
        span.set_status(StatusCode.OK)
        return result


async def run_pipeline(medication_text: str) -> None:
    setup_tracing()

    async with phoenix_mcp_session():
        await ensure_prompt_synced()
        historical_weaknesses_display, historical_weaknesses_prompt, historical_runs = await get_historical_weaknesses()
        _PIPELINE_CACHE.clear()
        _PIPELINE_CACHE["historical_weaknesses"] = historical_weaknesses_prompt
        _PIPELINE_CACHE["historical_runs"] = historical_runs

        tracer = otel_trace.get_tracer("drug_interaction_agent")

        with tracer.start_as_current_span("drug_interaction_pipeline") as root_span:
            root_span.set_attribute("openinference.span.kind", "CHAIN")
            root_span.set_attribute("input.value", medication_text)

            drugs_json = await _run_with_span(
                tracer, "resolve_all_medications",
                resolve_all_medications(medication_text),
                {"input.medication_text": medication_text},
            )

            interactions_json, _ = await asyncio.gather(
                _run_with_span(
                    tracer, "check_all_interactions",
                    check_all_interactions(drugs_json),
                    {"input.drugs_json": drugs_json},
                ),
                _run_with_span(
                    tracer, "get_fda_enrichment",
                    get_fda_enrichment(drugs_json),
                    {"input.drugs_json": drugs_json},
                ),
            )

            report = await _run_with_span(
                tracer, "synthesize_safety_report",
                synthesize_safety_report(drugs_json, interactions_json),
                {"input.drug_count": drugs_json.count('"name"')},
            )

            eval_json = await _run_with_span(
                tracer, "evaluate_report_quality",
                evaluate_report_quality(report),
                {"input.report_chars": len(report)},
            )

            output = await _run_with_span(
                tracer, "apply_self_improvement",
                apply_self_improvement(report),
                {"input.eval_scores": eval_json},
            )

            root_span.set_attribute("output.value", _PIPELINE_CACHE.get("report", output or ""))
            root_span.set_status(StatusCode.OK)

        # apply_self_improvement is authoritative — always prefer its cached report
        final = _PIPELINE_CACHE.get("report", output)

    _scores = _PIPELINE_CACHE.get("eval_scores", [])
    _score_str = (
        ", ".join(
            f"{e['section'].split()[0]}: "
            f"{e.get('overall', e.get('accuracy', 0)):.2f}"
            for e in _scores
        )
        if _scores
        else "see Phoenix dashboard"
    )
    _historical_runs = _PIPELINE_CACHE.get("historical_runs", historical_runs)
    _weak_display = historical_weaknesses_display if historical_weaknesses_display else "none"
    _judge_version = _PIPELINE_CACHE.get("judge_prompt_version", "built-in")

    if _OBSERVABILITY_HEADER not in final:
        final = final.rstrip()
        final += (
            f"\n\nPIPELINE OBSERVABILITY: Phoenix Project: drug-interaction-agent, "
            f"Judge Criteria: {_judge_version}, "
            f"Historical Runs: {_historical_runs}, "
            f"Weak Sections Detected: {_weak_display}, "
            f"Tools Traced: resolve_all_medications → check_all_interactions ∥ "
            f"get_fda_enrichment → synthesize_safety_report → evaluate_report_quality → "
            f"apply_self_improvement, "
            f"Eval Scores: {_score_str}\n"
        )

    print(final, end="", flush=True)
    if not final.endswith("\n"):
        print()


def main() -> None:
    msg = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "I take metformin, lisinopril, and ibuprofen"
    )
    asyncio.run(run_pipeline(msg))


if __name__ == "__main__":
    main()
