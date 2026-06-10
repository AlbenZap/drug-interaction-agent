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

"""Phoenix Experiments — benchmark the full pipeline against a fixed test dataset.

Runs all 7 benchmark drug combinations through the pipeline and records
per-section quality scores as a Phoenix Experiment. Each run creates a named
experiment version so you can compare scores across pipeline changes in
the Phoenix UI.

Usage:
    make experiment
    make experiment NAME="gemini-2.5-flash-v2"
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

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
from phoenix_client import (
    ensure_prompt_synced,
    get_historical_weaknesses,
    get_or_create_benchmark_dataset,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Pipeline task — runs on each benchmark example
# ---------------------------------------------------------------------------


async def _pipeline_task(example_input: dict) -> dict:
    """Run the full 6-tool pipeline on a single drug combination.

    Called by Phoenix for each example in the benchmark dataset.
    Returns structured output including the report and per-section scores.
    """
    _PIPELINE_CACHE.clear()
    _, weak_prompt, runs = await get_historical_weaknesses()
    _PIPELINE_CACHE["historical_weaknesses"] = weak_prompt
    _PIPELINE_CACHE["historical_runs"] = runs

    medication_text = example_input.get("medication_text", "")

    drugs_json = await resolve_all_medications(medication_text)
    interactions_json = await check_all_interactions(drugs_json)
    await get_fda_enrichment(drugs_json)
    report = await synthesize_safety_report(drugs_json, interactions_json)
    eval_json = await evaluate_report_quality(report)
    final_report = await apply_self_improvement(report)

    eval_scores = _PIPELINE_CACHE.get("eval_scores", [])
    section_scores = {
        e["section"]: round(e.get("overall", 0.5), 3) for e in eval_scores
    }
    drugs = _PIPELINE_CACHE.get("drugs", [])
    interactions = _PIPELINE_CACHE.get("interactions", [])

    return {
        "report": final_report,
        "section_scores": section_scores,
        "drugs_resolved": sum(1 for d in drugs if d.get("resolved")),
        "drugs_total": len(drugs),
        "interactions_found": len(interactions),
    }


# ---------------------------------------------------------------------------
# Evaluators — extract per-section scores from task output
# ---------------------------------------------------------------------------


def _section_evaluator(section_name: str):
    """Return an evaluator function for a specific report section."""
    def evaluator(output: dict) -> float:
        return output.get("section_scores", {}).get(section_name, 0.0)
    evaluator.__name__ = section_name.lower().replace(" ", "_").replace("&", "and")
    return evaluator


def eval_overall_quality(output: dict) -> float:
    scores = output.get("section_scores", {}).values()
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def eval_interactions_found(output: dict, metadata: dict) -> float:
    """Score 1.0 if interactions were found when expected, 0.0 otherwise."""
    expected = metadata.get("expected_severity", "none")
    found = output.get("interactions_found", 0)
    if expected == "none":
        return 1.0 if found == 0 else 0.5
    return 1.0 if found > 0 else 0.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    setup_tracing()
    asyncio.run(ensure_prompt_synced())

    experiment_name = sys.argv[1] if len(sys.argv) > 1 else f"pipeline-{date.today()}"

    print("Setting up benchmark dataset...")
    dataset = get_or_create_benchmark_dataset()
    print(f"Dataset: {dataset.name} ({len(dataset)} examples)")

    from phoenix_client import _get_client
    client = _get_client()

    evaluators = [
        eval_overall_quality,
        _section_evaluator("MEDICATION SUMMARY"),
        _section_evaluator("INTERACTION ALERTS"),
        _section_evaluator("FDA LABEL WARNINGS"),
        _section_evaluator("REAL-WORLD ADVERSE EVENTS"),
        _section_evaluator("RECOMMENDATIONS"),
        eval_interactions_found,
    ]

    print(f"Running experiment: {experiment_name}")
    print(f"This will call the full pipeline on {len(dataset)} drug combinations.\n")

    result = client.experiments.run_experiment(
        dataset=dataset,
        task=_pipeline_task,
        evaluators=evaluators,
        experiment_name=experiment_name,
        experiment_description=(
            f"Full pipeline benchmark — resolve → check_interactions → fda → "
            f"synthesize → evaluate → self_improve. "
            f"Judge criteria: druginteractionqualitycriteria (Phoenix Prompt Management)."
        ),
        experiment_metadata={
            "model": "gemini-2.5-flash",
            "pipeline_version": "v1",
        },
        print_summary=True,
    )

    # Print Phoenix URL for the experiment
    try:
        url = client.experiments.get_experiment_url(experiment_id=result["experiment_id"])
        print(f"\nView in Phoenix: {url}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
