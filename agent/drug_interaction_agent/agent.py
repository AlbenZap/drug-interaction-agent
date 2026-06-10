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

"""ADK agent definition for the Drug Interaction Monitor."""

from __future__ import annotations

import os
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from dotenv import load_dotenv

from instrumentation import setup_tracing
from drug_interaction_agent.prompts import SYSTEM_PROMPT
from drug_interaction_agent.tools import (
    apply_self_improvement,
    check_all_interactions,
    evaluate_report_quality,
    get_fda_enrichment,
    resolve_all_medications,
    synthesize_safety_report,
)

# Ensure env and tracing are set up when ADK CLI imports this module directly
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
setup_tracing()

_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Phoenix MCP — gives the agent tools for dataset history, prompt management, and
# connectivity verification at runtime. timeout=60 covers both npx/npm startup and
# individual Phoenix Cloud API calls (default 5 s is too short for either).
# Only attached when env vars are set.
_phoenix_api_key = os.environ.get("PHOENIX_API_KEY", "").strip()
_phoenix_base_url = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "").strip()

_phoenix_mcp: McpToolset | None = None
if _phoenix_api_key and _phoenix_base_url:
    _phoenix_mcp = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    "@arizeai/phoenix-mcp@latest",
                    "--baseUrl",
                    _phoenix_base_url,
                    "--apiKey",
                    _phoenix_api_key,
                ],
            ),
            timeout=60.0,
        ),
        # get-dataset-examples: read historical eval scores from past runs to guide
        #   synthesis toward historically weak sections (cross-run self-improvement).
        # add-dataset-examples: write this run's scores to Phoenix after completion,
        #   growing the history that future runs will learn from.
        # get-prompt-by-identifier: fetch versioned judge criteria from Phoenix
        #   Prompt Management so evaluation standards can be updated without redeploying.
        # list-projects: confirm Phoenix connectivity, get project ID.
        # list-traces / get-spans are excluded — they embed full LLM outputs in
        # span attributes, returning ~1.6M chars that overflow the context window.
        tool_filter=[
            "get-dataset-examples",
            "add-dataset-examples",
            "get-prompt-by-identifier",
            "list-projects",
        ],
    )

_tools = [
    FunctionTool(func=resolve_all_medications),
    FunctionTool(func=check_all_interactions),
    FunctionTool(func=get_fda_enrichment),
    FunctionTool(func=synthesize_safety_report),
    FunctionTool(func=evaluate_report_quality),
    FunctionTool(func=apply_self_improvement),
]
if _phoenix_mcp is not None:
    _tools.append(_phoenix_mcp)

root_agent = Agent(
    model=_model,
    name="drug_interaction_agent",
    instruction=SYSTEM_PROMPT,
    tools=_tools,
)
