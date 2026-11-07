"""NVIDIA-hosted model integration (optional backend).

Uses the OpenAI-compatible endpoint at integrate.api.nvidia.com with a
minimal ``requests`` implementation, so no OpenAI SDK is required. If
``langchain-openai`` is installed, :func:`longloom.init_llm("nvidia", ...)`
prefers that instead.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

__all__ = ["NVIDIAChatModel", "NVIDIA_BASE_URL", "MODELS"]

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Real model IDs on integrate.api.nvidia.com (verified via GET /v1/models).
# Context windows are intentionally conservative defaults; override
# ``context_window`` awareness in your processor config if you know better.
MODELS = {
    "nvidia/nemotron-3-ultra-550b-a55b": 8192,      # reasoning model
    "nvidia/nemotron-3-super-120b-a12b": 8192,      # reasoning model
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": 8192,
    "nvidia/llama-3.1-nemotron-70b-instruct": 8192,
    "nvidia/llama-3.1-nemotron-51b-instruct": 8192,
}
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


class NVIDIAChatModel(BaseChatModel):
    model_name: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    base_url: str = NVIDIA_BASE_URL
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    timeout: float = 120.0
    # None  → send nothing (model default behaviour, works on every model);
    # False → disable chain-of-thought ("chat_template_kwargs": {"thinking": false});
    # True  → request reasoning. Reasoning tokens come out of max_tokens!
    thinking: Optional[bool] = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.api_key:
            self.api_key = (
                os.environ.get("NVIDIA_API_KEY")
                or os.environ.get("NVIDIA_NIM_API_KEY")
                or os.environ.get("NEMOTRON_API_KEY")
            )
        if not self.api_key:
            raise ValueError(
                "NVIDIA API key not found. Set NVIDIA_API_KEY (get one at "
                "https://integrate.api.nvidia.com/) or pass api_key=... directly."
            )

    @property
    def _llm_type(self) -> str:
        return "nvidia-chat"

    @property
    def context_window(self) -> int:
        return MODELS.get(self.model_name, 8192)

    def _convert(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        role_map = {
            SystemMessage: "system",
            HumanMessage: "user",
            AIMessage: "assistant",
        }
        out = []
        for msg in messages:
            role = role_map.get(type(msg), "user")
            out.append({"role": role, "content": msg.content})
        return out

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: List[BaseMessage], stream: bool, stop: Optional[List[str]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._convert(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": stream,
        }
        if stop:
            payload["stop"] = stop
        if self.thinking is not None:
            # Reasoning-capable Nemotron models accept this toggle; passing it
            # on models that don't would error, so only send when requested.
            payload["chat_template_kwargs"] = {"thinking": bool(self.thinking)}
        return payload

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import requests

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=self._payload(messages, stream=False, stop=stop),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        msg = data["choices"][0]["message"]
        # Reasoning models may return content=None when reasoning consumed the
        # whole max_tokens budget — never propagate None downstream.
        content = msg.get("content") or ""
        message = AIMessage(content=content)
        generation_info = {
            "finish_reason": data["choices"][0].get("finish_reason"),
            "reasoning_tokens": len(msg.get("reasoning_content") or ""),
        }
        return ChatResult(
            generations=[ChatGeneration(message=message, generation_info=generation_info)]
        )

    def stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        import requests

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=self._payload(messages, stream=True, stop=stop),
            stream=True,
            timeout=self.timeout,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            body = decoded[6:]
            if body.strip() == "[DONE]":
                break
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content"):  # skip reasoning-only deltas / None
                    yield ChatGeneration(
                        message=AIMessage(content=delta["content"]),
                        generation_info={"finish_reason": choices[0].get("finish_reason")},
                    )