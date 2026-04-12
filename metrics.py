"""
metrics.py — Evaluation metrics (no third-party dependencies).

GenAI disclosure: Assistive tools (e.g. Cursor/LLM-based coding assistants) were used for
refactoring, documentation, and boilerplate. All changes were reviewed and tested locally.
"""

import re
import string


def _normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace, split into words."""
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return text.split()


def wer(references: list[str], hypotheses: list[str]) -> float:
    """Word Error Rate averaged over a list of (reference, hypothesis) pairs.

    WER = (S + D + I) / N  where N = total reference words.
    Computed via word-level Levenshtein distance.
    Returns 0.0 if all references are empty.
    """
    total_edits = 0
    total_ref_words = 0

    for ref, hyp in zip(references, hypotheses):
        r = _normalise(ref)
        h = _normalise(hyp)
        total_ref_words += len(r)
        total_edits += _levenshtein(r, h)

    return total_edits / total_ref_words if total_ref_words > 0 else 0.0


def _levenshtein(r: list[str], h: list[str]) -> int:
    """Word-level Levenshtein distance (substitutions, deletions, insertions)."""
    n, m = len(r), len(h)
    # dp[i][j] = edit distance between r[:i] and h[:j]
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if r[i - 1] == h[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]
