"""Normalize free-form text-prompt lists for SAM 3 inference."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    """Flattened SAM 3 queries plus the class index each query belongs to."""

    query_words: list[str]
    query_idx: list[int]
    class_prompts: list[str]

    @property
    def num_queries(self) -> int:
        return len(self.query_words)

    @property
    def num_classes(self) -> int:
        return len(self.class_prompts)


def _split_synonyms(item: str | Sequence[str]) -> list[str]:
    if isinstance(item, str):
        parts = [part.strip() for part in item.split(",")]
    elif isinstance(item, Sequence):
        parts = []
        for nested in item:
            if not isinstance(nested, str):
                raise TypeError(
                    f"Prompt synonym groups must contain strings, got {type(nested)!r}"
                )
            parts.extend(_split_synonyms(nested))
    else:
        raise TypeError(
            "Each text prompt must be a string or a sequence of synonym strings, "
            f"got {type(item)!r}"
        )

    names = [part.replace("\n", "").strip() for part in parts if part.strip()]
    if not names:
        raise ValueError("Encountered an empty text prompt after stripping whitespace")
    return names


def normalize_text_prompts(text_prompts: Iterable[str | Sequence[str]]) -> PromptSpec:
    """Convert a user prompt list into per-query strings and class indices.

    Supported forms (can be mixed):

    * ``["building", "road"]`` — one SAM 3 query per class
    * ``["building,house", "road"]`` — comma-separated synonyms of one class
    * ``[["building", "house"], "road"]`` — explicit synonym groups
    """
    if text_prompts is None:
        raise ValueError("text_prompts is required and must be a non-empty list")

    if isinstance(text_prompts, str):
        raise TypeError(
            "text_prompts must be a list of prompts, not a single string. "
            "Use ['building', 'road'] instead of 'building'."
        )

    items = list(text_prompts)
    if not items:
        raise ValueError("text_prompts must contain at least one class prompt")

    query_words: list[str] = []
    query_idx: list[int] = []
    class_prompts: list[str] = []

    for class_id, item in enumerate(items):
        synonyms = _split_synonyms(item)
        class_prompts.append(",".join(synonyms))
        query_words.extend(synonyms)
        query_idx.extend([class_id] * len(synonyms))

    return PromptSpec(
        query_words=query_words,
        query_idx=query_idx,
        class_prompts=class_prompts,
    )
