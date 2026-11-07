"""Shared helpers for the longloom examples."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXIT_OK, EXIT_FAIL, EXIT_SKIPPED = 0, 1, 3


def guard(requirement: str, ok: bool) -> None:
    """Exit with SKIP status unless *ok* — examples degrade gracefully."""
    if not ok:
        print(f"SKIPPED: {requirement}")
        sys.exit(EXIT_SKIPPED)


def env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


import os  # noqa: E402  (used by env())


SAMPLE_DOC = """
AI in healthcare report. Medical imaging reaches dermatologist-level accuracy
with FDA-cleared devices. Drug discovery is accelerated by AlphaFold and
graph neural networks. Personalized medicine uses genomics and polygenic
risk scores. Clinical decision support warns of sepsis early. Federated
learning protects data privacy while transformers and foundation models
power multimodal care. FDA oversight addresses algorithmic bias. COVID-19
sped adoption; democratization and economic impact drive recommendations.
""" * 12


class DemoLLM:
    """Deterministic zero-dependency LLM stand-in so examples run offline."""

    def __call__(self, prompt: str) -> str:
        if "CHUNK RESULTS:" in prompt:
            # Deliberately long (~500 tokens) so small-window examples
            # genuinely exercise the repair path.
            return (
                "## Summary\nAI in healthcare spans medical imaging, drug discovery "
                "(AlphaFold, graph neural networks), personalized medicine, clinical "
                "decision support for sepsis, federated learning for data privacy, "
                "transformers and foundation models for multimodal care, FDA oversight "
                "of algorithmic bias, COVID-19 lessons, and democratization with real "
                "economic impact driving recommendations." * 8
            )
        if "CURRENT RESULT:" in prompt:
            return "Condensed: medical imaging, drug discovery, AlphaFold, personalized " \
                   "medicine, sepsis, federated learning, data privacy, FDA, " \
                   "algorithmic bias, multimodal."
        return "[chunk ok] medical imaging, drug discovery, personalized medicine."


def show(title: str, result: dict, processor) -> None:
    print(f"\n=== {title} ===")
    print(f"  chunks={result['num_chunks']}  repair={result['repair_attempts']}  "
          f"input_tokens={result['total_input_tokens']:,}")
    hits = processor.search_session(result["session_id"], "AlphaFold")
    print(f"  search 'AlphaFold': {len(hits)} chunk(s)")
    print(f"  result preview: {result['final_result'][:140]}...")