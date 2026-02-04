"""Fuzzy name matcher for contact profile merge suggestions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import NamedTuple, Sequence

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


# =============================================================================
# Types
# =============================================================================


@dataclass(frozen=True, slots=True)
class MatchConfig:
    """Matching configuration."""

    token_threshold: float = 85.0
    suggestion_threshold: float = 70.0
    top_n: int = 3

    weight_quality: float = 0.45
    weight_query_coverage: float = 0.35
    weight_candidate_coverage: float = 0.20

    penalty_per_missed: float = 0.12
    penalty_floor: float = 0.20
    penalty_reverse: float = 0.97
    penalty_first_mismatch: float = 0.50

    bonus_first: float = 1.00
    bonus_last: float = 1.10
    bonus_subset: float = 1.03

    score_exact: float = 100.0
    score_equivalent: float = 95.0
    score_initial: float = 90.0
    score_normalized: float = 92.0


class TokenMatch(NamedTuple):
    """A matched token pair."""

    query_token: str
    candidate_token: str
    query_idx: int
    candidate_idx: int
    similarity: float


@dataclass(frozen=True, slots=True)
class Candidate:
    """A candidate with one or more name variants."""

    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Result of matching a query against a candidate."""

    candidate: Candidate
    matched_name: str
    score: float
    matched_tokens: tuple[TokenMatch, ...]


class _IndexedMatch(NamedTuple):
    """Internal: token match by index only."""

    query_idx: int
    candidate_idx: int
    similarity: float


# =============================================================================
# Normalization (module-level, cacheable)
# =============================================================================

_ARABIC_PREFIXES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("abdul", re.compile(r"\b(abdel|abd el|abd al|abd-el|abd-al)\b", re.I)),
    ("al", re.compile(r"\b(el|al)[-\s]?(?=\w)", re.I)),
    ("abu", re.compile(r"\b(abo|abou)\b", re.I)),
    ("bin", re.compile(r"\b(ben|ibn)\b", re.I)),
    ("bint", re.compile(r"\bbent\b", re.I)),
)
_WHITESPACE: re.Pattern[str] = re.compile(r"\s+")
_REPEAT: re.Pattern[str] = re.compile(r"(.)\1{3,}")
_CHAR_MAP: dict[int, str] = str.maketrans({"y": "i", "e": "a", "o": "u"})
_DOUBLES: tuple[str, ...] = ("mm", "dd", "ss", "ll", "aa")


@cache
def _normalize(name: str) -> str:
    """Normalize name: lowercase, Arabic prefixes, whitespace."""
    if not name:
        return ""
    result = name.lower().strip()
    for replacement, pattern in _ARABIC_PREFIXES:
        result = pattern.sub(replacement, result)
    result = _REPEAT.sub(r"\1\1", result)
    result = result.replace("-", " ")
    return _WHITESPACE.sub(" ", result).strip()


@cache
def _tokenize(name: str) -> tuple[str, ...]:
    """Split normalized name into tokens."""
    return tuple(_normalize(name).split()) if name else ()


@cache
def _simplify(token: str) -> str:
    """Simplify Arabic letter variants (y→i, e→a, o→u, doubles)."""
    result = token.translate(_CHAR_MAP)
    for double in _DOUBLES:
        result = result.replace(double, double[0])
    return result


# =============================================================================
# Similarity (module-level, cacheable)
# =============================================================================


@cache
def _compute_similarity(
    token1: str,
    token2: str,
    equivalents: frozenset[tuple[str, str]],
    config: MatchConfig,
) -> float:
    """Compute similarity between two tokens (0-100)."""
    if token1 == token2:
        return config.score_exact

    # Check equivalence table
    equiv1 = next((v for k, v in equivalents if k == token1), None)
    equiv2 = next((v for k, v in equivalents if k == token2), None)
    if equiv1 and equiv2:
        return config.score_equivalent if equiv1 == equiv2 else 0.0

    # Initial matching (e.g., "J" matches "John")
    if len(token1) == 1 and len(token2) >= 2 and token2[0] == token1:
        return config.score_initial
    if len(token2) == 1 and len(token1) >= 2 and token1[0] == token2:
        return config.score_initial

    # Simplified form matching
    simplified1, simplified2 = _simplify(token1), _simplify(token2)
    if simplified1 == simplified2:
        return config.score_normalized

    # Fuzzy matching
    score = max(
        JaroWinkler.similarity(token1, token2) * 100,
        fuzz.ratio(token1, token2),
    )
    if simplified1 != token1 or simplified2 != token2:
        score = max(score, fuzz.ratio(simplified1, simplified2))
    return float(score)


# =============================================================================
# Matcher
# =============================================================================


class NameMatcher:
    """Fuzzy name matcher for contact merge suggestions."""

    def __init__(
        self,
        config: MatchConfig | None = None,
        equivalents: dict[str, str] | None = None,
    ) -> None:
        self.config = config or MatchConfig()
        self._equivalents_frozen = frozenset((equivalents or {}).items())

    def find_matches(
        self,
        query: str,
        candidates: Sequence[Candidate],
        threshold: float | None = None,
        top_n: int | None = None,
    ) -> list[MatchResult]:
        """Find candidates matching the query, sorted by score descending."""
        threshold = threshold if threshold is not None else self.config.suggestion_threshold
        top_n = top_n if top_n is not None else self.config.top_n

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        results = [
            result
            for candidate in candidates
            if (result := self._match_candidate(query_tokens, candidate, threshold))
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n]

    def _similarity(self, token1: str, token2: str) -> float:
        """Compute token similarity (0-100), using cached module function."""
        return _compute_similarity(token1, token2, self._equivalents_frozen, self.config)

    # -------------------------------------------------------------------------
    # Matching
    # -------------------------------------------------------------------------

    def _match_candidate(
        self,
        query_tokens: tuple[str, ...],
        candidate: Candidate,
        threshold: float,
    ) -> MatchResult | None:
        """Match query against a candidate, returning best match if above threshold."""
        best_score = 0.0
        best_name = ""
        best_matches: list[_IndexedMatch] = []

        for name in candidate.names:
            score, matches = self._match_bidirectional(query_tokens, _tokenize(name))
            if score > best_score:
                best_score, best_name, best_matches = score, name, matches

        if best_score < threshold:
            return None

        candidate_tokens = _tokenize(best_name)
        return MatchResult(
            candidate=candidate,
            matched_name=best_name,
            score=best_score,
            matched_tokens=tuple(
                TokenMatch(
                    query_token=query_tokens[match.query_idx],
                    candidate_token=candidate_tokens[match.candidate_idx],
                    query_idx=match.query_idx,
                    candidate_idx=match.candidate_idx,
                    similarity=match.similarity,
                )
                for match in best_matches
            ),
        )

    def _match_bidirectional(
        self,
        query_tokens: tuple[str, ...],
        candidate_tokens: tuple[str, ...],
    ) -> tuple[float, list[_IndexedMatch]]:
        """Match tokens in both directions, return best result."""
        config = self.config

        forward_score, forward_matches = self._match_sequential(query_tokens, candidate_tokens)
        if len(forward_matches) == len(candidate_tokens) < len(query_tokens):
            forward_score *= config.bonus_subset

        reverse_score, reverse_matches = self._match_sequential(candidate_tokens, query_tokens)
        reverse_score *= config.penalty_reverse
        if len(reverse_matches) == len(query_tokens) < len(candidate_tokens):
            reverse_score *= config.bonus_subset

        if self._similarity(query_tokens[0], candidate_tokens[0]) < config.token_threshold:
            forward_score *= config.penalty_first_mismatch
            reverse_score *= config.penalty_first_mismatch

        if forward_score >= reverse_score:
            return forward_score, forward_matches

        # Flip indices for reverse matches
        flipped = [
            _IndexedMatch(match.candidate_idx, match.query_idx, match.similarity)
            for match in reverse_matches
        ]
        return reverse_score, flipped

    def _match_sequential(
        self,
        query_tokens: tuple[str, ...],
        candidate_tokens: tuple[str, ...],
    ) -> tuple[float, list[_IndexedMatch]]:
        """Match query tokens to candidate tokens in sequential order."""
        config = self.config
        matches: list[_IndexedMatch] = []
        search_start = 0

        for query_idx, query_token in enumerate(query_tokens):
            best_candidate_idx, best_similarity = -1, 0.0

            for cand_idx in range(search_start, len(candidate_tokens)):
                similarity = self._similarity(query_token, candidate_tokens[cand_idx])
                if similarity >= config.token_threshold and similarity > best_similarity:
                    best_candidate_idx, best_similarity = cand_idx, similarity
                    if similarity == config.score_exact:
                        break

            if best_candidate_idx >= 0:
                matches.append(_IndexedMatch(query_idx, best_candidate_idx, best_similarity))
                search_start = best_candidate_idx + 1

        return self._compute_score(matches, len(query_tokens), len(candidate_tokens)), matches

    def _compute_score(
        self,
        matches: list[_IndexedMatch],
        query_len: int,
        candidate_len: int,
    ) -> float:
        """Compute overall match score from individual token matches."""
        if not matches:
            return 0.0

        config = self.config

        # Single-token query: simpler scoring
        if query_len == 1:
            match = matches[0]
            is_first_position = match.candidate_idx == 0
            is_strong_match = match.similarity >= config.score_equivalent
            factor = 1.0 if is_first_position and is_strong_match else 0.9
            return match.similarity * factor

        # Multi-token: weighted combination of quality and coverage
        avg_similarity = sum(match.similarity for match in matches) / len(matches)
        score = (
            avg_similarity * config.weight_quality
            + (len(matches) / query_len) * 100 * config.weight_query_coverage
            + (len(matches) / candidate_len) * 100 * config.weight_candidate_coverage
        )

        # Penalty for missed query tokens
        missed_count = query_len - len(matches)
        score *= max(config.penalty_floor, 1.0 - missed_count * config.penalty_per_missed)

        # Bonuses for matching first/last tokens
        matched_query_indices = {match.query_idx for match in matches}
        matched_candidate_indices = {match.candidate_idx for match in matches}

        if 0 in matched_query_indices and 0 in matched_candidate_indices:
            score *= config.bonus_first
        if (query_len - 1) in matched_query_indices and (candidate_len - 1) in matched_candidate_indices:
            score *= config.bonus_last

        return score


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import json
    from pathlib import Path

    test_data = json.loads(Path("./test_cases.json").read_text())

    for test_case in test_data:
        candidates = [Candidate(names=tuple(c["names"])) for c in test_case["candidates"]]
        matcher = NameMatcher()

        for query in test_case["queries"]:
            print("=" * 80)
            print(f"Query: {query}")
            print(f"Candidates: {[', '.join(c.names) for c in candidates]}")
            for rank, result in enumerate(matcher.find_matches(query, candidates), 1):
                tokens = [
                    (token_match.query_token, token_match.candidate_token, f"{token_match.similarity:.0f}%")
                    for token_match in result.matched_tokens
                ]
                print(f"  {rank}. [{result.score:5.1f}] {result.matched_name} — {tokens}")
