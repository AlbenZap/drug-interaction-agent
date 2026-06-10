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

"""Phoenix OTel tracing setup.

Requires PHOENIX_API_KEY and PHOENIX_COLLECTOR_ENDPOINT in the environment.
PHOENIX_PROJECT_NAME defaults to "drug-interaction-agent".
"""

from __future__ import annotations

import os

_provider = None


def setup_tracing():
    """Register the Phoenix OTel tracer provider once. No-op if already registered or API key missing."""
    global _provider
    if _provider is not None:
        return _provider
    if not os.environ.get("PHOENIX_API_KEY", "").strip():
        return None
    from phoenix.otel import register
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
    _provider = register(
        project_name=os.environ.get("PHOENIX_PROJECT_NAME", "drug-interaction-agent"),
        batch=False,
        auto_instrument=False,
        verbose=False,
    )
    GoogleGenAIInstrumentor().instrument(tracer_provider=_provider)
    return _provider
