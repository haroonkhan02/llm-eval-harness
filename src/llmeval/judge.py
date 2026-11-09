"""Pairwise LLM-as-judge: prompt, verdict parser and swap-and-aggregate.

The judge sees one user question (or a two-turn conversation) and two assistant answers
and ends its reply with ``[[A]]``, ``[[B]]`` or ``[[C]]`` (tie). To control position bias,
every pair is judged twice with the answers swapped; the final verdict is kept only when
both orders agree, otherwise it becomes a tie and is flagged as inconsistent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from llmeval.backends import Backend, Completion

SYSTEM_PROMPT = (
    "You are an impartial reviewer comparing two AI assistants' answers to the same user "
    "request. Judge helpfulness, correctness, depth and how well each answer follows the "
    "user's instructions. Do not let the order of the answers, their length or the "
    "assistants' names influence you. Explain your reasoning briefly, then give your "
    "verdict on the last line in exactly this form: [[A]] if assistant A is better, "
    "[[B]] if assistant B is better, or [[C]] for a tie."
)

_VERDICT = re.compile(r"\[\[([ABC])\]\]")


def _turns(conversation: list[dict]) -> list[tuple[str, str]]:
    """(user, assistant) pairs from an alternating message list."""
    msgs = list(conversation)
    expected = ("user", "assistant")
    if len(msgs) % 2 or any(m["role"] != expected[i % 2] for i, m in enumerate(msgs)):
        raise ValueError("conversation must alternate user/assistant, starting with user")
    return [(msgs[i]["content"], msgs[i + 1]["content"]) for i in range(0, len(msgs), 2)]


def build_prompt(conv_a: list[dict], conv_b: list[dict], turn: int) -> str:
    """User message for judging ``turn`` (1 or 2) of two conversations."""
    ta, tb = _turns(conv_a), _turns(conv_b)
    if turn not in (1, 2) or len(ta) < turn or len(tb) < turn:
        raise ValueError(f"turn {turn} not available")
    if [u for u, _ in ta[:turn]] != [u for u, _ in tb[:turn]]:
        raise ValueError("the two conversations ask different questions")
    if turn == 1:
        q = ta[0][0]
        return (f"[User request]\n{q}\n\n[Assistant A's answer]\n{ta[0][1]}\n\n"
                f"[Assistant B's answer]\n{tb[0][1]}\n\n"
                "Compare the two answers and give your verdict.")
    parts = []
    for label, t in (("A", ta), ("B", tb)):
        parts.append(f"<<< Conversation with assistant {label} >>>\n"
                     f"[User]\n{t[0][0]}\n[Assistant {label}]\n{t[0][1]}\n"
                     f"[User]\n{t[1][0]}\n[Assistant {label}]\n{t[1][1]}\n")
    return ("\n".join(parts) + "\nCompare only the two assistants' answers to the SECOND "
            "user message, in the context of the whole conversation, and give your verdict.")


def parse_verdict(text: str) -> str | None:
    """Return 'A', 'B' or 'C' from the last [[X]] tag, or None if there is none."""
    found = _VERDICT.findall(text or "")
    return found[-1] if found else None


@dataclass
class PairJudgment:
    """Verdicts in terms of the original A/B order."""

    forward: str | None            # verdict with (A, B) as given
    backward: str | None           # verdict with answers swapped, mapped back to A/B
    completions: list[Completion] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return self.forward is not None and self.forward == self.backward

    @property
    def final(self) -> str | None:
        """Swap-and-aggregate: agree -> that verdict; disagree -> tie; unparsable -> None."""
        if self.forward is None or self.backward is None:
            return None
        return self.forward if self.consistent else "C"

    @property
    def winner(self) -> str | None:
        """MT-Bench style label: model_a, model_b, tie or tie (inconsistent)."""
        f = self.final
        if f is None:
            return None
        if not self.consistent:
            return "tie (inconsistent)"
        return {"A": "model_a", "B": "model_b", "C": "tie"}[f]


_SWAP = {"A": "B", "B": "A", "C": "C"}


def judge_pair(backend: Backend, conv_a: list[dict], conv_b: list[dict], turn: int,
               swap: bool = True) -> PairJudgment:
    """Judge one pair; with ``swap`` the reversed order is judged too."""
    c1 = backend.complete(SYSTEM_PROMPT, build_prompt(conv_a, conv_b, turn))
    fwd = None if c1.refused else parse_verdict(c1.text)
    if not swap:
        return PairJudgment(forward=fwd, backward=fwd, completions=[c1])
    c2 = backend.complete(SYSTEM_PROMPT, build_prompt(conv_b, conv_a, turn))
    bwd_raw = None if c2.refused else parse_verdict(c2.text)
    bwd = _SWAP[bwd_raw] if bwd_raw else None
    return PairJudgment(forward=fwd, backward=bwd, completions=[c1, c2])
