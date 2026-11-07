"""A deterministic, phase-aware fake chat model for provider-agnostic tests.

Detects which workflow phase a prompt belongs to (map / merge / repair) the
same way the real prompts are shaped, and returns canned responses. No
network, no API keys — proves the graph is LLM-agnostic.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional

from pydantic import Field
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

MAP_RE = re.compile(r"CHUNK \d+/\d+:")
REPAIR_RE = re.compile(r"REPAIR|CONDENSE|TOO LONG|CURRENT RESULT:|MERGED RESULT:", re.IGNORECASE)
MERGE_RE = re.compile(r"CHUNK RESULTS:|combine these into", re.IGNORECASE)

KEY_CONCEPTS = [
    "AI in healthcare", "medical imaging", "drug discovery",
    "personalized medicine", "clinical decision support",
    "federated learning", "transformers", "graph neural networks",
    "data privacy", "algorithmic bias", "FDA", "sepsis",
    "diabetic retinopathy", "COVID-19", "AlphaFold",
    "foundation models", "multimodal", "democratization",
    "economic impact", "recommendations",
]


class PhaseAwareFakeChatModel(BaseChatModel):
    map_response: str = "[MAP RESULT] ok"
    merge_response: str = "[MERGE RESULT] " + "; ".join(KEY_CONCEPTS)
    repair_response: str = "[REPAIRED RESULT] " + "; ".join(KEY_CONCEPTS)
    calls: List[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "phase-aware-fake"

    def _detect_phase(self, prompt: str) -> str:
        if MAP_RE.search(prompt):
            return "map"
        if REPAIR_RE.search(prompt):
            return "repair"
        if MERGE_RE.search(prompt):
            return "merge"
        return "unknown"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        phase = self._detect_phase(prompt)
        self.calls.append(phase)
        response = {
            "map": self.map_response,
            "merge": self.merge_response,
            "repair": self.repair_response,
        }.get(phase, "[UNKNOWN PHASE] " + prompt[:80])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    @property
    def phase_counts(self) -> dict:
        return {p: self.calls.count(p) for p in ("map", "merge", "repair")}


def make_doc(concepts: List[str], repeats: int = 30) -> str:
    """Synthetic long document embedding every tracked concept."""
    paragraph = (
        "Comprehensive analysis of AI in healthcare. Medical imaging reaches "
        "dermatologist-level accuracy. Drug discovery is accelerated by "
        "AlphaFold and graph neural networks. Personalized medicine uses "
        "genomics. Clinical decision support warns of sepsis early. "
        "Federated learning protects data privacy while transformers and "
        "foundation models power multimodal care. FDA oversight addresses "
        "algorithmic bias. COVID-19 sped adoption; democratization and "
        "economic impact drive recommendations.\n\n"
    )
    return paragraph * repeats