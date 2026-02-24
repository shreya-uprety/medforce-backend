"""
LLM utility functions — truncation safety and retry wrapper.

Shared by intake_agent, clinical_agent, and monitoring_agent.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("gateway.agents.llm_utils")


def is_response_complete(text: str) -> bool:
    """
    Check if an LLM response appears complete (not truncated).

    Returns False if text is empty, ends mid-word, or has other
    truncation indicators.
    """
    if not text or not text.strip():
        return False

    stripped = text.strip()

    # Ends mid-word: last char is a letter and there's no sentence-ending punctuation nearby
    if len(stripped) < 10:
        return True  # Very short responses are likely complete

    # Check for obvious truncation: ends with a space followed by an incomplete word
    # or ends without sentence-ending punctuation after a long response
    last_char = stripped[-1]

    # Ends with common truncation indicators
    truncation_indicators = ["...", "—", "–"]
    if any(stripped.endswith(ind) for ind in truncation_indicators):
        # Ellipsis at end of a long response is suspicious
        if len(stripped) > 50:
            return False

    # If text ends mid-sentence (no terminal punctuation) and is long,
    # check if it ends mid-word
    if last_char.isalpha() and len(stripped) > 100:
        # Check if it looks like a cut-off sentence
        # A properly terminated response usually ends with punctuation
        last_sentence = stripped.split(".")[-1].strip()
        if len(last_sentence) > 60 and not any(
            last_sentence.endswith(p) for p in ["?", "!", ".", ")", "]", '"']
        ):
            return False

    return True


async def llm_generate(
    client: Any,
    model: str,
    contents: str,
    max_retries: int = 2,
    critical: bool = False,
) -> str | None:
    """
    Call the LLM with retry and exponential backoff.

    P4 enhanced:
    - Default: 2 retries (3 total attempts) for most calls
    - Critical paths (risk scoring, emergency detection): use critical=True
      for 3 retries (4 total attempts) with longer backoff
    - Exponential backoff: 0.5s, 1.0s, 2.0s (critical: 1.0s, 2.0s, 4.0s)
    - All callers have deterministic fallback logic for total failure

    Returns the response text, or None if exhausted so callers
    use their existing fallback.
    """
    effective_retries = max_retries if not critical else max(max_retries, 3)
    base_backoff = 1.0 if critical else 0.5

    for attempt in range(effective_retries + 1):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
            )
            text = response.text
            if isinstance(text, str) and text.strip():
                return text
            # Empty response — treat as transient failure
            logger.warning("LLM returned empty response (attempt %d)", attempt + 1)
        except Exception as exc:
            logger.warning(
                "LLM call failed (attempt %d/%d): %s",
                attempt + 1, effective_retries + 1, exc,
            )

        if attempt < effective_retries:
            backoff = base_backoff * (2 ** attempt)
            await asyncio.sleep(backoff)

    if effective_retries > 0:
        logger.error(
            "LLM call exhausted all %d attempts — returning None",
            effective_retries + 1,
        )
    return None
