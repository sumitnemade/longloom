#!/usr/bin/env python3
"""07 — Ultimate provider-agnosticism: a bare Python function as the LLM.

Shows the minimum contract: a callable (str -> str) is a complete LLM.
Useful for wrapping in-house gateways, batching, caching, or routing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import SAMPLE_DOC, show

from longloom import InMemoryContextStore, MapReduceProcessor


class RoutingLLM:
    """Pretend multi-model gateway: cheap model for maps, big model for merge."""

    def __init__(self):
        self.calls = {"map": 0, "merge": 0, "repair": 0}

    def __call__(self, prompt: str) -> str:
        if "CURRENT RESULT:" in prompt:                       # repair → "big" model
            self.calls["repair"] += 1
            return "Condensed summary: medical imaging, drug discovery, AlphaFold, " \
                   "personalized medicine, sepsis, federated learning, data privacy, " \
                   "FDA, algorithmic bias, multimodal."
        if "CHUNK RESULTS:" in prompt:                        # merge → "big" model
            self.calls["merge"] += 1
            return ("## Merged analysis\nmedical imaging, drug discovery (AlphaFold, "
                    "graph neural networks), personalized medicine, clinical decision "
                    "support (sepsis), federated learning, data privacy, transformers, "
                    "foundation models, multimodal care, FDA oversight, algorithmic "
                    "bias, COVID-19, democratization, economic impact, recommendations."
                    * 5)  # long enough to exceed the tiny window below
        self.calls["map"] += 1                                # map → "cheap" model
        return json.dumps({"chunk": "ok",
                           "topics": ["medical imaging", "drug discovery",
                                      "personalized medicine"]})


def main() -> int:
    llm = RoutingLLM()
    processor = MapReduceProcessor(
        llm=llm,                       # instance with __call__ — still just a callable
        store=InMemoryContextStore(),
        max_chunk_tokens=600,
        max_context_tokens=200,        # force the repair path
    )
    result = processor.process(SAMPLE_DOC, task="Summarize key themes",
                               session_id="callable-demo")
    show("Callable LLM (multi-model routing)", result, processor)
    print(f"  call routing: {llm.calls}")
    assert llm.calls["merge"] == 1 and llm.calls["repair"] >= 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())