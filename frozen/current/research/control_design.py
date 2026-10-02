"""Pure stopping-rule references for the common frozen-classifier comparison.

This module does not load data/models, fit anything, predict hidden features,
or time a pipeline. Probability construction stays in certificate_kernel:
choose_probability(probability_lower, probability_upper, method=0). Production
timing should compile equivalent checks in the same backend for every method;
do not charge one method Python reference overhead but another compiled checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


RULES = ("class_preserving", "fixed_prefix", "confidence", "entropy")
CONFIDENCE_GRID = (.50, .60, .70, .80, .90, .95, .975, .99, .995, 1.0)
NORMALIZED_ENTROPY_GRID = (.01, .025, .05, .10, .15, .20, .30, .40, .50, .70, 1.0)


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str
    needs_exact: bool = False
    certified_class: int = -1
    diagnostic: float | None = None


def checkpoint_ends(total_blocks: int, batch_size: int = 4) -> tuple[int, ...]:
    """Same nonempty checkpoint schedule as FrozenForest.prepare_batches."""
    if not isinstance(total_blocks, (int, np.integer)) or total_blocks < 1:
        raise ValueError("total_blocks must be a positive integer")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return tuple(range(int(batch_size), int(total_blocks), int(batch_size))) + (int(total_blocks),)


def canonical_prefix(prefix_blocks: int, ends: Sequence[int]) -> int:
    """Round a requested prefix upward to an actual shared checkpoint."""
    e = tuple(int(x) for x in ends)
    if not e or any(x < 1 for x in e) or any(a >= b for a, b in zip(e, e[1:])):
        raise ValueError("ends must be positive and strictly increasing")
    if not isinstance(prefix_blocks, (int, np.integer)) or not 1 <= prefix_blocks <= e[-1]:
        raise ValueError("prefix_blocks must lie in [1,total_blocks]")
    return next(x for x in e if x >= prefix_blocks)


def validate_block_order(order: Sequence[int], total_blocks: int) -> None:
    """The shared order covers each real feature-processing block exactly once."""
    o = np.asarray(order)
    if o.shape != (total_blocks,) or not np.issubdtype(o.dtype, np.integer):
        raise ValueError("order must contain total_blocks integer block IDs")
    if not np.array_equal(np.sort(o), np.arange(total_blocks)):
        raise ValueError("order must be a permutation of all block IDs")


def _probability(probability: Sequence[float]) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("probability must be a finite class vector")
    if np.any(p < 0.0) or np.any(p > 1.0) or not np.isclose(p.sum(), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("probability must lie in the simplex; it is not normalized here")
    return p


def certified_class(score_lower: Sequence[float], score_upper: Sequence[float]) -> int:
    """Strict outward-guarded logit separation; -1 means not established.

    For every reachable complete score z, lower <= z <= upper. If lower[w]
    exceeds every competing upper, the complete model's unique argmax is w.
    Ties, nonfinite/invalid intervals fail closed. Bounds are not narrowed here.
    """
    lo = np.asarray(score_lower, dtype=np.float64)
    hi = np.asarray(score_upper, dtype=np.float64)
    if lo.ndim != 1 or len(lo) < 2 or hi.shape != lo.shape:
        raise ValueError("score bounds must be matching class vectors")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
        return -1
    winner = int(np.argmax(lo))
    for c in range(len(lo)):
        if c != winner and not lo[winner] > hi[c]:
            return -1
    return winner


def normalized_entropy(probability: Sequence[float]) -> float:
    """H(p)/log(C), with 0 log 0 = 0; no model or label is consulted."""
    p = _probability(probability)
    positive = p > 0.0
    value = float(-np.dot(p[positive], np.log(p[positive])) / np.log(len(p)))
    return min(1.0, max(0.0, value))


def decide_stop(
    rule: str,
    processed_blocks: int,
    total_blocks: int,
    *,
    probability: Sequence[float] | None = None,
    score_lower: Sequence[float] | None = None,
    score_upper: Sequence[float] | None = None,
    prefix_blocks: int | None = None,
    confidence_threshold: float | None = None,
    entropy_threshold: float | None = None,
    collapsed: bool = False,
) -> StopDecision:
    """Call only at shared checkpoints; return decision, never construct p.

    class_preserving also checks argmax(p)==certified class because the outer
    probability box used for minimax construction may include unrealizable
    vectors. No rule in this function uses a Brier certificate or delta.
    Terminal decisions request the exact compiled full-model probability.
    """
    if rule not in RULES:
        raise ValueError("Unknown rule: " + str(rule))
    if not isinstance(total_blocks, (int, np.integer)) or total_blocks < 1:
        raise ValueError("total_blocks must be a positive integer")
    if not isinstance(processed_blocks, (int, np.integer)) or not 0 <= processed_blocks <= total_blocks:
        raise ValueError("Invalid processed_blocks")
    if rule == "fixed_prefix" and (
        not isinstance(prefix_blocks, (int, np.integer)) or not 1 <= prefix_blocks <= total_blocks
    ):
        raise ValueError("fixed_prefix requires a canonical prefix_blocks")
    if rule == "confidence" and (
        confidence_threshold is None or not np.isfinite(confidence_threshold)
        or not 0.0 <= confidence_threshold <= 1.0
    ):
        raise ValueError("confidence requires a threshold in [0,1]")
    if rule == "entropy" and (
        entropy_threshold is None or not np.isfinite(entropy_threshold)
        or not 0.0 <= entropy_threshold <= 1.0
    ):
        raise ValueError("entropy requires a normalized threshold in [0,1]")
    if collapsed:
        return StopDecision(True, "reachable_scores_collapsed", needs_exact=True)
    if processed_blocks == total_blocks:
        return StopDecision(True, "complete_prefix", needs_exact=True)
    if rule == "fixed_prefix":
        return StopDecision(processed_blocks >= prefix_blocks, "fixed_prefix" if processed_blocks >= prefix_blocks else "continue")
    if rule == "class_preserving":
        if score_lower is None or score_upper is None:
            raise ValueError("class_preserving requires score bounds")
        winner = certified_class(score_lower, score_upper)
        if winner < 0:
            return StopDecision(False, "class_not_separated")
        if probability is None:
            # Caller can calculate minimax p only after this cheap prerequisite.
            return StopDecision(False, "minimax_probability_required", certified_class=winner)
        p = _probability(probability)
        if len(p) != len(np.asarray(score_lower)):
            raise ValueError("Probability and score class counts differ")
        agrees = int(np.argmax(p)) == winner
        return StopDecision(agrees, "class_preserved" if agrees else "minimax_argmax_disagrees", certified_class=winner)
    if probability is None:
        raise ValueError(rule + " requires the current minimax probability")
    p = _probability(probability)
    if rule == "confidence":
        value = float(np.max(p))
        stop = value >= confidence_threshold
        return StopDecision(stop, "confidence" if stop else "continue", diagnostic=value)
    value = normalized_entropy(p)
    stop = value <= entropy_threshold
    return StopDecision(stop, "entropy" if stop else "continue", diagnostic=value)


def select_development_setting(
    summaries: Iterable[Mapping],
    *,
    tolerance: float,
    risk_key: str,
    time_key: str = "mean_complete_ms",
    expected_setting_ids: Sequence[str] | None = None,
) -> dict | None:
    """Deterministic empirical development selection, not a risk guarantee.

    The protocol must choose risk_key before running the sweep (e.g. mean
    observed-label half-Brier excess, or max subject-mean excess). Summaries
    must come from the same entire development cohort and complete online
    timing. This helper cannot establish those provenance requirements itself.
    Returns None if no candidate satisfies the development constraint.
    """
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    rows = [dict(x) for x in summaries]
    ids = [str(x["setting_id"]) for x in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate setting_id")
    if expected_setting_ids is not None:
        expected = [str(x) for x in expected_setting_ids]
        if len(set(expected)) != len(expected) or set(ids) != set(expected):
            raise ValueError("Incomplete or unexpected development sweep")
    eligible = []
    for row in rows:
        risk, elapsed = float(row[risk_key]), float(row[time_key])
        if not np.isfinite(risk) or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Nonfinite risk/time or negative complete time")
        if risk <= tolerance:
            eligible.append(row)
    if not eligible:
        return None
    return min(eligible, key=lambda x: (float(x[time_key]), float(x[risk_key]), str(x["setting_id"])))


__all__ = ["RULES", "CONFIDENCE_GRID", "NORMALIZED_ENTROPY_GRID", "StopDecision",
           "checkpoint_ends", "canonical_prefix", "validate_block_order", "certified_class",
           "normalized_entropy", "decide_stop", "select_development_setting"]
