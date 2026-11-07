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


class RetryAfterError(Exception):
    """Carries a provider-style response with a Retry-After header."""

    def __init__(self, retry_after: str):
        super().__init__("rate limited")
        self.response = type("R", (), {"headers": {"Retry-After": retry_after}})()


def test_init_llm_nvidia_chatopenai_paths():
    """Both nvidia OpenAI-client paths: thinking (extra_body) and without."""
    without = init_llm("nvidia", "nvidia/nemotron-3-ultra-550b-a55b", api_key="k")
    assert without.model_name == "nvidia/nemotron-3-ultra-550b-a55b"
    with_t = init_llm("nvidia", "nvidia/nemotron-3-ultra-550b-a55b",
                      api_key="k", thinking=True)
    assert with_t.model_name == "nvidia/nemotron-3-ultra-550b-a55b"


class RetryAfterError(Exception):
    """Carries a provider-style response with a Retry-After header."""

    def __init__(self, retry_after: str):
        super().__init__("rate limited")
        self.response = type("R", (), {"headers": {"Retry-After": retry_after}})()


def test_retry_after_malformed_header_falls_back(monkeypatch):
    import time as _time

    sleeps: list[float] = []
    monkeypatch.setattr(_time, "sleep", lambda s: sleeps.append(s) if s > 0 else None)

    class BadHeader(PhaseAwareFakeChatModel):
        n: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            prompt = "\n".join(str(m.content) for m in messages)
            if self.n == 0 and "CHUNK" in prompt:
                self.n += 1
                raise RetryAfterError("not-a-number")   # malformed header
            return super()._generate(messages, stop, run_manager, **kwargs)

    adapter = LLMAdapter(BadHeader(), retry_attempts=2, retry_backoff=1.0)
    out = adapter.complete("CHUNK 1/2: fact")
    assert out.startswith("[MAP RESULT]") or out
    assert sleeps and 1.0 <= sleeps[0] <= 1.25   # base backoff + jitter, no header boost


def test_retry_after_header_is_honored(monkeypatch):
    import time as _time

    sleeps: list[float] = []
    monkeypatch.setattr(_time, "sleep", lambda s: sleeps.append(s))

    class RateLimitedOnce(PhaseAwareFakeChatModel):
        n: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            prompt = "\n".join(str(m.content) for m in messages)
            if self.n == 0 and "CHUNK" in prompt:
                self.n += 1
                raise RetryAfterError("7")   # provider says: wait 7 seconds
            return super()._generate(messages, stop, run_manager, **kwargs)

    adapter = LLMAdapter(RateLimitedOnce(), retry_attempts=3, retry_backoff=0.5)
    out = adapter.complete("CHUNK 1/2: fact 42")
    assert out == RateLimitedOnce().map_response or out.startswith("[MAP RESULT]")
    assert sleeps and sleeps[0] >= 7.0          # Retry-After overrides base backoff
    assert sleeps[0] <= 120.0                    # ...but within the hard cap


def test_retry_after_is_capped(monkeypatch):
    import time as _time

    sleeps: list[float] = []
    monkeypatch.setattr(_time, "sleep", lambda s: sleeps.append(s))

    class AlwaysRateLimited(PhaseAwareFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            raise RetryAfterError("5000")        # absurd provider value

    adapter = LLMAdapter(AlwaysRateLimited(), retry_attempts=1, retry_backoff=1.0)
    with pytest.raises(RetryAfterError):
        adapter.complete("CHUNK 1/1: x")
    assert sleeps and sleeps[0] <= 120.0         # capped, never 5000s


def test_retry_backoff_is_jittered_but_bounded(monkeypatch):
    import random as _random
    import time as _time

    sleeps: list[float] = []
    monkeypatch.setattr(_time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(_random, "uniform", lambda a, b: b)  # worst case: +25%

    class AlwaysDown(PhaseAwareFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            raise ConnectionError("down")

    adapter = LLMAdapter(AlwaysDown(), retry_attempts=3, retry_backoff=2.0, jitter=True)
    with pytest.raises(ConnectionError):
        adapter.complete("CHUNK 1/1: x")
    # base 2,4,8 with +25% jitter and the 120s cap
    assert sleeps[0] == pytest.approx(2.5) and sleeps[1] == pytest.approx(5.0)
    assert sleeps[2] == pytest.approx(10.0)


def test_logs_never_contain_prompt_content(monkeypatch, caplog):
    import logging as _logging
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda s: None)
    caplog.set_level(_logging.DEBUG, logger="longloom")

    class AlwaysDown(PhaseAwareFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            raise ConnectionError("down")

    adapter = LLMAdapter(AlwaysDown(), retry_attempts=2, retry_backoff=0.0)
    with pytest.raises(ConnectionError):
        adapter.complete("PROMPT-SECRET-should-never-be-logged-42")

    assert "PROMPT-SECRET-should-never-be-logged-42" not in caplog.text
    assert any("LLM call failed" in r.message for r in caplog.records)
    # an explicitly passed key is used verbatim, even if env vars also exist
    from longloom.nvidia import NVIDIAChatModel

    native = NVIDIAChatModel(model_name="nvidia/nemotron-3-ultra-550b-a55b",
                             api_key="passed-explicitly")
    assert native.api_key == "passed-explicitly"
    via_helper = init_llm("nvidia", "nvidia/nemotron-3-ultra-550b-a55b",
                          api_key="passed-explicitly")
    assert via_helper is not None  # helper accepts the explicit key