"""Provider-agnostic LLM access.

The workflow never talks to a vendor SDK directly — it goes through
:class:`LLMAdapter`, which accepts:

* any LangChain ``BaseChatModel`` (OpenAI, Anthropic, NVIDIA, Groq, Ollama,
  vLLM, FakeChatModel in tests, ...),
* a plain callable ``str -> str``,
* or a provider/model pair via :func:`init_llm` convenience.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

__all__ = ["LLMLike", "LLMAdapter", "init_llm", "DEFAULT_NVIDIA_MODEL"]

LLMLike = Union[BaseChatModel, Callable[[str], str]]

try:  # keep the real default in one place (longloom.nvidia)
    from .nvidia import DEFAULT_MODEL as DEFAULT_NVIDIA_MODEL
except Exception:  # pragma: no cover - defensive
    DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


class LLMAdapter:
    """Normalizes any supported LLM form behind a single ``complete()`` call.

    Includes retry with exponential backoff for transient API failures
    (429 rate limits, timeouts, 5xx) — a multi-hour map/merge run must not
    die on one blip. ``min_interval`` optionally paces successive calls to
    stay under provider rate limits during long matrix runs.
    """

    def __init__(self, llm: LLMLike, retry_attempts: int = 4, retry_backoff: float = 3.0,
                 min_interval: float = 0.0):
        if isinstance(llm, BaseChatModel):
            self._model: Optional[BaseChatModel] = llm
            self._fn: Optional[Callable[[str], str]] = None
        elif callable(llm):
            self._model = None
            self._fn = llm
        else:
            raise TypeError(
                "llm must be a langchain BaseChatModel or a callable (str -> str); "
                f"got {type(llm).__name__}. Use longloom.init_llm(...) to build one "
                "from a provider/model pair."
            )
        self.retry_attempts = max(0, retry_attempts)
        self.retry_backoff = max(0.0, retry_backoff)
        self.min_interval = max(0.0, min_interval)
        self._last_call = 0.0

    def _once(self, prompt: str) -> str:
        if self._model is not None:
            response = self._model.invoke([HumanMessage(content=prompt)])
            content = response.content
            return content if isinstance(content, str) else str(content)
        assert self._fn is not None
        return self._fn(prompt)

    def complete(self, prompt: str) -> str:
        import time

        if self.min_interval > 0:
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
        last_exc: Optional[Exception] = None
        for attempt in range(self.retry_attempts + 1):
            try:
                result = self._once(prompt)
                self._last_call = time.time()
                return result
            except Exception as exc:  # transient API failures included
                last_exc = exc
                self._last_call = time.time()
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_backoff * (2 ** attempt))
        assert last_exc is not None
        raise last_exc

    @property
    def backend(self) -> str:
        if self._model is not None:
            return f"langchain:{type(self._model).__name__}"
        return "callable"


def init_llm(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseChatModel:
    """Build a chat model from a provider name — convenience helper.

    * ``"nvidia"`` → NVIDIA-hosted models (OpenAI-compatible endpoint).
    * any other provider string is forwarded to LangChain's
      ``init_chat_model`` when the matching integration is installed
      (e.g. ``"openai"``, ``"anthropic"``, ``"groq"``, ``"ollama"``).
    """
    provider = provider.lower()
    if provider == "nvidia":
        resolved_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("NEMOTRON_API_KEY")
        if resolved_key is None:
            # Raise the friendly, NVIDIA-specific error.
            from .nvidia import NVIDIAChatModel

            NVIDIAChatModel(model_name=model or DEFAULT_NVIDIA_MODEL)  # raises ValueError

        # `thinking` is a Nemotron-specific extension the OpenAI client path
        # can't express cleanly — route to our native backend when requested.
        thinking = kwargs.pop("thinking", None)
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            from .nvidia import NVIDIAChatModel

            return NVIDIAChatModel(
                model_name=model or DEFAULT_NVIDIA_MODEL,
                api_key=resolved_key,
                thinking=thinking,
                **kwargs,
            )

        if thinking is not None:
            # extra_body (not model_kwargs): the OpenAI SDK's typed signature
            # rejects unknown top-level kwargs, but extra_body goes into the
            # JSON body verbatim — exactly what NVIDIA expects.
            return ChatOpenAI(
                model=model or DEFAULT_NVIDIA_MODEL,
                api_key=resolved_key,
                base_url=kwargs.pop("base_url", "https://integrate.api.nvidia.com/v1"),
                extra_body={"chat_template_kwargs": {"thinking": bool(thinking)}},
                **kwargs,
            )
        return ChatOpenAI(
            model=model or DEFAULT_NVIDIA_MODEL,
            api_key=resolved_key,
            base_url=kwargs.pop("base_url", "https://integrate.api.nvidia.com/v1"),
            **kwargs,
        )

    # Direct integration imports — works with just langchain-core + the one
    # provider package installed (no `langchain` meta-package required).
    direct = {
        "openai": ("langchain_openai", "ChatOpenAI"),
        "anthropic": ("langchain_anthropic", "ChatAnthropic"),
        "groq": ("langchain_groq", "ChatGroq"),
        "ollama": ("langchain_ollama", "ChatOllama"),
        "google_genai": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
        "mistralai": ("langchain_mistralai", "ChatMistralAI"),
        "together": ("langchain_openai", "ChatOpenAI"),
        "deepseek": ("langchain_deepseek", "ChatDeepSeek"),
    }
    if provider in direct:
        module_name, class_name = direct[provider]
        try:
            module = __import__(module_name, fromlist=[class_name])
        except ImportError as exc:
            raise ImportError(
                f"Provider {provider!r} needs {module_name}: "
                f"pip install {module_name.replace('_', '-')}"
            ) from exc
        cls = getattr(module, class_name)
        return cls(model=model, api_key=api_key, **kwargs)

    # Anything else: LangChain's provider registry (needs the meta-package).
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:
        raise ImportError(
            f"Provider {provider!r} has no direct integration mapped. Install "
            "its langchain-<provider> package, or the 'langchain' meta-package. "
            "Alternatively pass a ready BaseChatModel instance to MapReduceProcessor."
        ) from exc
    return init_chat_model(model, model_provider=provider, api_key=api_key, **kwargs)