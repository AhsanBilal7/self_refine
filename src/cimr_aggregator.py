"""Answer aggregation over the N refined rollouts (Algorithm 1, lines 10-13).

Given the terminal answers ``a_i = Extract(r_i^(D))``, two predictions are
returned side by side:

1. ``majority`` -- the method's prediction: plurality vote
   ``a = Vote({a_i})`` over normalised answers. Empty answers do not vote.
   Ties on vote count are broken by the larger summed PRM score of the tied
   group, then by the lower first rollout index. Without a verifier
   (``--no_prm``) every PRM score is 0 and ties fall back to rollout order.
2. ``max_prm`` -- a diagnostic: the answer of the rollout whose terminal trace
   has the highest PRM score. It is not part of the method.

Grouping uses ``normalize_answer_key``, a light string canonicalisation.
Mathematical equivalence (e.g. ``1/2`` vs ``0.5``) is only resolved later,
when each prediction is compared against the gold answer.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def normalize_answer_key(ans: str) -> str:
    """Canonical voting key for an extracted answer.

    Removes spaces and LaTeX spacing commands (``\\,``, ``\\!``, ``\\;``),
    unwraps one enclosing pair of braces and strips trailing decimal zeros
    (``3.500`` -> ``3.5``). Case is preserved.
    """
    s = str(ans or "").strip()
    if not s:
        return ""
    s = s.replace(" ", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    if s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s



def aggregate_rollouts(
    rollouts: List[Dict[str, Any]],
    gold_answer: Optional[str] = None,
    ev=None,
    data_name: str = "math",
) -> Dict[str, Any]:
    """Aggregate rollout records into the majority-vote and max-PRM predictions.

    Parameters
    ----------
    rollouts    : rollout records from ``cimr_rollout._build_rollout_record``.
    gold_answer : gold answer; correctness fields are ``None`` without it.
    ev          : dataset evaluator used for the correctness check.
    data_name   : dataset name forwarded to the correctness check.

    Returns
    -------
    dict with ``majority_pred``, ``majority_is_correct``,
    ``majority_rollout_idx``, ``majority_vote_count``, ``max_prm_pred``,
    ``max_prm_is_correct``, ``max_prm_rollout_idx``, ``max_prm_score``,
    ``per_rollout_summary`` and ``n_rollouts``.
    """
    if not rollouts:
        return dict(
            majority_pred="", majority_is_correct=None,
            majority_rollout_idx=-1, majority_vote_count=0,
            max_prm_pred="", max_prm_is_correct=None,
            max_prm_rollout_idx=-1, max_prm_score=0.0,
            per_rollout_summary=[], n_rollouts=0,
        )

    per_rollout: List[Dict[str, Any]] = [
        dict(
            rollout_idx=int(r.get("rollout_idx", -1)),
            final_answer=str(r.get("final_answer", "")),
            final_prm_score=float(r.get("final_prm_score", 0.0)),
        )
        for r in rollouts
    ]

    # ── Plurality vote ──────────────────────────────────────────────────────
    bucket: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for r in per_rollout:
        key = normalize_answer_key(r["final_answer"])
        if not key:
            continue
        if key not in bucket:
            bucket[key] = dict(
                key=key,
                display=r["final_answer"].strip(),
                count=0,
                sum_prm=0.0,
                first_idx=int(r["rollout_idx"]),
                rollout_idxs=[],
            )
        bucket[key]["count"] += 1
        bucket[key]["sum_prm"] += float(r["final_prm_score"])
        bucket[key]["rollout_idxs"].append(int(r["rollout_idx"]))

    if bucket:
        # Rank by (votes desc, summed PRM desc, first rollout index asc).
        winner = sorted(
            bucket.values(),
            key=lambda b: (b["count"], b["sum_prm"], -b["first_idx"]),
            reverse=True,
        )[0]
        majority_pred = winner["display"]
        majority_rollout_idx = int(winner["first_idx"])
        majority_vote_count = int(winner["count"])
    else:
        majority_pred = ""
        majority_rollout_idx = -1
        majority_vote_count = 0

    # ── Max-PRM selection (diagnostic) ──────────────────────────────────────
    best = max(per_rollout, key=lambda r: r["final_prm_score"])
    max_prm_pred = best["final_answer"]
    max_prm_rollout_idx = int(best["rollout_idx"])
    max_prm_score = float(best["final_prm_score"])

    def _is_correct(pred: str) -> Optional[bool]:
        if gold_answer is None or ev is None or not pred:
            return None
        try:
            from .inference_methods import _check_correct
            return bool(_check_correct(pred, gold_answer, ev, data_name))
        except Exception as e:
            logger.debug(f"[CIMR aggregator] correctness check failed: {e}")
            return None

    majority_is_correct = _is_correct(majority_pred)
    max_prm_is_correct = _is_correct(max_prm_pred)

    logger.debug(
        f"[CIMR aggregator] "
        f"majority={majority_pred!r} (correct={majority_is_correct}, "
        f"votes={majority_vote_count})  "
        f"max_prm={max_prm_pred!r} (correct={max_prm_is_correct}, "
        f"score={max_prm_score:.4f})"
    )

    return dict(
        majority_pred=majority_pred,
        majority_is_correct=majority_is_correct,
        majority_rollout_idx=majority_rollout_idx,
        majority_vote_count=majority_vote_count,
        max_prm_pred=max_prm_pred,
        max_prm_is_correct=max_prm_is_correct,
        max_prm_rollout_idx=max_prm_rollout_idx,
        max_prm_score=max_prm_score,
        per_rollout_summary=per_rollout,
        n_rollouts=len(rollouts),
    )


__all__ = ["aggregate_rollouts", "normalize_answer_key"]
