"""Model backends behind one small interface: ``complete(system, user) -> Completion``.

- ``FakeBackend``: deterministic, for tests and pipeline smoke runs.
- ``AnthropicBackend``: Claude via the Anthropic SDK. The client is injectable so tests
  never touch the network. Requires an Anthropic API key at run time.
- ``MLXBackend``: an open-weight model running locally on Apple silicon via ``mlx-lm``
  (greedy decoding, no API cost). The loader is injectable so tests need no model.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

# USD per million tokens (input, output), Anthropic list prices.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
          "claude-haiku-5-5": (0.10, 0.50)}


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    stop_reason: str = "end_turn"
    cost_usd: float = 0.0

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    def as_dict(self) -> dict:
        return asdict(self)


class Backend(Protocol):
    name: str

    def complete(self, system: str, user: str) -> Completion: ...


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


class FakeBackend:
    """Returns scripted replies: a fixed list (cycled) or a function of the prompt."""

    def __init__(self, replies: Sequence[str] | Callable[[str, str], str], name: str = "fake"):
        self.name = name
        self._replies = replies
        self._i = 0
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        if callable(self._replies):
            text = self._replies(system, user)
        else:
            text = self._replies[self._i % len(self._replies)]
            self._i += 1
        return Completion(text=text, model=self.name,
                          input_tokens=len((system + user).split()),
                          output_tokens=len(text.split()))


class AnthropicBackend:
    """Claude judge. Thinking is adaptive (always on for Opus 5.5); depth via ``effort``.

    Server-side refusal fallbacks are enabled; a refusal that survives the fallback chain
    comes back as ``stop_reason == "refusal"`` with empty text, never as a verdict.
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low",
                 max_tokens: int = 4000, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}:{effort}"

    def complete(self, system: str, user: str) -> Completion:
        t0 = time.perf_counter()
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        latency = time.perf_counter() - t0
        stop = resp.stop_reason or "end_turn"
        text = "" if stop == "refusal" else "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = resp.usage
        served_by = getattr(resp, "model", None) or self.model
        return Completion(
            text=text, model=served_by,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            latency_s=latency, stop_reason=stop,
            cost_usd=cost_usd(served_by, usage.input_tokens, usage.output_tokens),
        )


class MLXBackend:
    """Local judge on Apple silicon. Greedy decoding, so a run is deterministic.

    ``loader`` and ``generator`` default to ``mlx_lm.load`` / ``mlx_lm.stream_generate``;
    tests pass fakes. Cost is reported as 0 (no API bill; the hardware is the cost).
    """

    def __init__(self, model: str, max_tokens: int = 512, loader=None, generator=None):
        if loader is None or generator is None:
            import mlx_lm

            loader = loader or mlx_lm.load
            generator = generator or mlx_lm.stream_generate
        self.model, self.tokenizer = loader(model)
        self._generate = generator
        self.model_id = model
        self.max_tokens = max_tokens
        self.name = f"mlx:{model.split('/')[-1]}"

    def _prompt(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def complete(self, system: str, user: str) -> Completion:
        prompt = self._prompt(system, user)
        t0 = time.perf_counter()
        text, last = [], None
        for last in self._generate(self.model, self.tokenizer, prompt,
                                   max_tokens=self.max_tokens):
            text.append(last.text)
        latency = time.perf_counter() - t0
        finish = getattr(last, "finish_reason", None) if last is not None else None
        return Completion(
            text="".join(text), model=self.model_id,
            input_tokens=int(getattr(last, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(last, "generation_tokens", 0) or 0),
            latency_s=latency,
            stop_reason="max_tokens" if finish == "length" else "end_turn",
        )
