from __future__ import annotations

from typing import Iterable, Tuple

import regex as re


def char_ngrams(s: str, n: int = 3) -> set:
    s = s or ""
    tokens = [s[i : i + n] for i in range(max(0, len(s) - n + 1))]
    return set(tokens)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / max(1, union)


def ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    return jaccard(char_ngrams(s1, n), char_ngrams(s2, n))


def edit_distance(s1: str, s2: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein  # type: ignore

        return int(Levenshtein.distance(s1, s2))
    except Exception:
        # simple DP fallback
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                cur = dp[j]
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[j] = min(
                    dp[j] + 1,      # delete
                    dp[j - 1] + 1,  # insert
                    prev + cost,    # replace
                )
                prev = cur
        return dp[n]
