"""LLMAdapter accepts every supported LLM form."""
from __future__ import annotations

import pytest

from longloom import LLMAdapter
from longloom.llm import init_llm

from fake_llm import PhaseAwareFakeChatModel


def test_callable_backend():
    adapter = LLMAdapter(lambda p: "OK:" + p[:10])
    assert adapter.backend == "callable"
    assert adapter.complete("hello world").startswith("OK:hello")


def test_basechatmodel_backend():
    fake = PhaseAwareFakeChatModel()
    adapter = LLMAdapter(fake)
    assert adapter.backend == "langchain:PhaseAwareFakeChatModel"
    out = adapter.complete("CHUNK 1/3:\ncontent {with} braces")
    assert out == fake.map_response


def test_rejects_invalid_llm():
    with pytest.raises(TypeError):
        LLMAdapter(42)  # type: ignore[arg-type]


def test_render_prompt_is_brace_safe():
    from longloom.prompts import render_prompt

    template = "Task: {task}\nContent: {chunk_content}"
    out = render_prompt(template, {"task": "t", "chunk_content": 'json {"a": 1} and {unknown}'})
    assert '{"a": 1} and {unknown}' in out  # content braces untouched
    assert "Task: t" in out


def test_init_llm_nvidia_requires_explicit_key():
    """LongLoom never reads keys from the environment — pass api_key=... ."""
    from longloom.nvidia import NVIDIAChatModel

    with pytest.raises(ValueError, match="never reads keys from the environment"):
        init_llm("nvidia", "nvidia/nemotron-3-ultra-550b-a55b", api_key=None)
    # an explicitly passed key is used verbatim, even if env vars also exist
    native = NVIDIAChatModel(model_name="nvidia/nemotron-3-ultra-550b-a55b",
                             api_key="passed-explicitly")
    assert native.api_key == "passed-explicitly"
    via_helper = init_llm("nvidia", "nvidia/nemotron-3-ultra-550b-a55b",
                          api_key="passed-explicitly")
    assert via_helper is not None  # helper accepts the explicit key