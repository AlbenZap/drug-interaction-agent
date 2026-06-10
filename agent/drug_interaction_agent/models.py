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

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, computed_field


class Drug(BaseModel):
    name: str
    rxcui: Optional[str] = None
    resolved: bool = False


class Interaction(BaseModel):
    drug1_name: str
    drug2_name: str
    severity: str  # "major" | "moderate" | "minor" | "unknown"
    description: str
    source: str = ""


class DrugLabel(BaseModel):
    rxcui: str
    drug_name: str
    boxed_warning: list[str] = Field(default_factory=list)
    warnings_and_cautions: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    drug_interactions: list[str] = Field(default_factory=list)


class AdverseEvent(BaseModel):
    drug_name: str
    total_reports: int = 0
    serious_count: int = 0
    top_reactions: list[str] = Field(default_factory=list)


class SectionEval(BaseModel):
    section: str
    accuracy: float = 0.0
    caution: float = 0.0
    clarity: float = 0.0
    citation: float = 0.0
    reasoning: str = ""

    @computed_field  # type: ignore[misc]
    @property
    def overall(self) -> float:
        return round((self.accuracy + self.caution + self.clarity + self.citation) / 4.0, 3)
