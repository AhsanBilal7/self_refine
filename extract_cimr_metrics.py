#!/usr/bin/env python3
"""
Refinement-dynamics metrics from breadth-depth refinement result stores.

Reads every ``cimr_full_results_<dataset>_N<N>_d<D>[<suffix>].json`` under
``--results_dir`` and writes one ``metrics_summary.json`` per model folder
under ``--output_dir`` (keyed by input filename). ``export_paper_csvs.py``
turns these into the CSVs behind the paper's refinement-dynamics figures.

Usage
-----
    python extract_cimr_metrics.py \
        --results_dir experiments_results/results_paired \
        --output_dir  experiments_results/cimr_metrics

Notation follows the paper's appendix "Diagnostic Metric Definitions". At depth
d, rollout i has the pre-correction answer alpha = Extract(g^(d))
(``reasoning_answer``) and the post-correction answer beta = Extract(r^(d))
(``corrected_answer``). Answers are compared after ``_norm``: whitespace and
LaTeX spacing removed, one brace layer unwrapped, trailing decimal zeros
stripped, lower-cased. Depth-wise votes use simple string matching against the
gold answer, so they can differ slightly from the final accuracy, which uses
the symbolic evaluator.

Metrics per file
----------------
accuracy
  majority_accuracy           final plurality-vote accuracy (stored flag)
  max_prm_accuracy            diagnostic max-PRM selection accuracy
  oracle_accuracy             pass@N: some rollout's final answer is correct
  pass_at_1_accuracy          mean per-rollout accuracy
  depth_wise_majority_acc     A(d): plurality-vote accuracy at depth d
  depth_improvement_rate      A(d) - A(d-1) for d >= 2; A(1) at d = 1
  depth_recovery_rate         rho_rec(d): wrong at d-1 -> correct at d (0 at d = 1)
  depth_regression_rate       rho_reg(d): correct at d-1 -> wrong at d (0 at d = 1)

correction_effectiveness (per depth, over all question-rollout pairs)
  answer_change_rate          mu_Delta(d): beta differs from alpha (beta non-empty)
  flip_wrong_to_correct       mu_{W->C}(d)
  flip_correct_to_wrong       mu_{C->W}(d)
  flip_wrong_to_wrong         changed but still wrong
  net_correction_benefit      B(d) = mu_{W->C}(d) - mu_{C->W}(d)

diversity
  mean_n_unique_final_answers mean |A_q^(D)|
  frac_questions_diverse      terminal diversity rate: |A_q^(D)| > 1
  depth_wise_answer_agreement U(d): all non-empty rollout answers agree
  mean_final_prm_std / mean_final_prm_range   spread of terminal PRM scores

Diagnostic PRM sections (empty if the run used --no_prm)
  prm_dynamics      per depth: PRM of g^(d) vs r^(d) and the lift from correction
  prm_calibration   PRM score of correct vs incorrect selections
  voting            vote margin and majority vs max-PRM agreement
  prm_monotonicity  whether r^(d) PRM scores rise across depths

efficiency
  mean_elapsed_sec_per_question, total_elapsed_sec, mean_elapsed_sec_per_rollout
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─── Answer normalisation (cimr_aggregator.normalize_answer_key + lower-casing) ─

def _norm(ans: str) -> str:
    s = str(ans or "").strip()
    if not s:
        return ""
    s = s.replace(" ", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    # Unwrap single surrounding braces
    if s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]
    # Strip trailing zeros from decimals: "3.500" → "3.5"
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s.lower()


def _simple_correct(pred: str, gold: str) -> Optional[bool]:
    """String-normalised correctness. Falls back to None when either arg is empty."""
    if not pred or not gold:
        return None
    return _norm(pred) == _norm(gold)


# ─── Step / depth helpers ────────────────────────────────────────────────────

def _step_at_depth(rollout: dict, depth: int) -> Optional[dict]:
    for s in (rollout.get("steps") or []):
        if int(s.get("step_idx", -1)) == depth:
            return s
    return None


def _depth_majority_vote(rollouts: List[dict], depth: int) -> str:
    """Return the majority-vote answer (normalised) across rollouts at a depth."""
    answers = []
    for r in rollouts:
        if not r:
            continue
        s = _step_at_depth(r, depth)
        if s:
            a = str(s.get("corrected_answer") or s.get("reasoning_answer") or "")
            if _norm(a):
                answers.append(_norm(a))
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def _safe_mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _safe_std(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    m = _safe_mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


# ─── Main extractor ───────────────────────────────────────────────────────────

def extract_metrics(store: dict, source_file: str) -> dict:
    questions = list(store.values())
    if not questions:
        return {"source_file": source_file, "n_questions": 0}

    sample = questions[0]
    N = int(sample.get("N", 0))
    D = int(sample.get("depth", 0))
    n_total = len(questions)

    # ── 1. Accuracy ──────────────────────────────────────────────────────────
    n_with_gold = sum(1 for q in questions if q.get("gold_answer"))
    n_majority_correct = sum(
        1 for q in questions
        if (q.get("aggregation") or {}).get("majority_is_correct") is True
    )
    n_maxprm_correct = sum(
        1 for q in questions
        if (q.get("aggregation") or {}).get("max_prm_is_correct") is True
    )

    # oracle: any rollout correct (simple norm)
    n_oracle = 0
    pass_at_1_values: List[float] = []
    for q in questions:
        gold = q.get("gold_answer") or ""
        if not gold:
            continue
        rollouts = [r for r in (q.get("rollouts") or []) if r]
        hits = [
            1 if _simple_correct(str(r.get("final_answer") or ""), gold) is True else 0
            for r in rollouts
        ]
        if any(h == 1 for h in hits):
            n_oracle += 1
        if hits:
            pass_at_1_values.append(sum(hits) / len(hits))

    # depth-wise majority accuracy + recovery/regression
    depth_majority_correct: Dict[int, int] = {}
    depth_recovery: Dict[int, int] = {}    # newly correct at depth d
    depth_regression: Dict[int, int] = {}  # newly wrong at depth d
    for d in range(1, D + 1):
        depth_majority_correct[d] = 0
        depth_recovery[d] = 0
        depth_regression[d] = 0

    for q in questions:
        gold = q.get("gold_answer") or ""
        if not gold:
            continue
        rollouts = [r for r in (q.get("rollouts") or []) if r]
        prev_correct = None
        for d in range(1, D + 1):
            pred_norm = _depth_majority_vote(rollouts, d)
            is_corr = (_norm(gold) == pred_norm) if (pred_norm and gold) else False
            if is_corr:
                depth_majority_correct[d] += 1
            if prev_correct is not None:
                if not prev_correct and is_corr:
                    depth_recovery[d] += 1
                elif prev_correct and not is_corr:
                    depth_regression[d] += 1
            prev_correct = is_corr

    denom_gold = max(n_with_gold, 1)
    denom_total = max(n_total, 1)

    accuracy = {
        "majority_accuracy": round(n_majority_correct / denom_total, 4),
        "max_prm_accuracy": round(n_maxprm_correct / denom_total, 4),
        "oracle_accuracy": round(n_oracle / denom_gold, 4),
        "pass_at_1_accuracy": round(_safe_mean(pass_at_1_values), 4),
        "depth_wise_majority_accuracy": {
            str(d): round(depth_majority_correct[d] / denom_gold, 4)
            for d in range(1, D + 1)
        },
        "depth_improvement_rate": {
            str(d): round(
                (depth_majority_correct[d] - depth_majority_correct.get(d - 1, depth_majority_correct[d]))
                / denom_gold, 4
            )
            for d in range(1, D + 1)
        },
        "depth_recovery_rate": {
            str(d): round(depth_recovery[d] / denom_gold, 4)
            for d in range(1, D + 1)
        },
        "depth_regression_rate": {
            str(d): round(depth_regression[d] / denom_gold, 4)
            for d in range(1, D + 1)
        },
    }
    # No A(0) is stored, so depth 1 reports A(1) itself.
    accuracy["depth_improvement_rate"]["1"] = round(
        depth_majority_correct.get(1, 0) / denom_gold, 4
    )

    # ── 2. PRM Dynamics per depth ────────────────────────────────────────────
    prm_dynamics: Dict[str, Any] = {}
    for d in range(1, D + 1):
        r_prms, c_prms, lifts = [], [], []
        n_improved = n_hurt = 0
        for q in questions:
            for r in (q.get("rollouts") or []):
                if not r:
                    continue
                s = _step_at_depth(r, d)
                if not s:
                    continue
                rp = float(s.get("reasoning_prm_score") or 0.0)
                cp = float(s.get("corrected_prm_score") or 0.0)
                r_prms.append(rp)
                c_prms.append(cp)
                lift = cp - rp
                lifts.append(lift)
                if lift > 0:
                    n_improved += 1
                elif lift < 0:
                    n_hurt += 1

        n = max(len(r_prms), 1)
        prm_dynamics[str(d)] = {
            "mean_reasoning_prm": round(_safe_mean(r_prms), 4),
            "mean_corrected_prm": round(_safe_mean(c_prms), 4),
            "mean_prm_lift": round(_safe_mean(lifts), 4),
            "prm_lift_std": round(_safe_std(lifts), 4),
            "frac_prm_improved": round(n_improved / n, 4),
            "frac_prm_hurt": round(n_hurt / n, 4),
            "n_rollout_steps": len(r_prms),
        }

    # ── 3. Correction Effectiveness per depth ────────────────────────────────
    correction_eff: Dict[str, Any] = {}
    for d in range(1, D + 1):
        n_changed = n_flip_w2c = n_flip_c2w = n_flip_w2w = n_steps = 0
        for q in questions:
            gold = q.get("gold_answer") or ""
            for r in (q.get("rollouts") or []):
                if not r:
                    continue
                s = _step_at_depth(r, d)
                if not s:
                    continue
                ra = str(s.get("reasoning_answer") or "")
                ca = str(s.get("corrected_answer") or "")
                n_steps += 1
                changed = _norm(ra) != _norm(ca) and bool(_norm(ca))
                if changed:
                    n_changed += 1
                    if gold:
                        ra_ok = _simple_correct(ra, gold) is True
                        ca_ok = _simple_correct(ca, gold) is True
                        if not ra_ok and ca_ok:
                            n_flip_w2c += 1
                        elif ra_ok and not ca_ok:
                            n_flip_c2w += 1
                        else:
                            n_flip_w2w += 1

        denom = max(n_steps, 1)
        correction_eff[str(d)] = {
            "answer_change_rate": round(n_changed / denom, 4),
            "flip_wrong_to_correct": round(n_flip_w2c / denom, 4),
            "flip_correct_to_wrong": round(n_flip_c2w / denom, 4),
            "flip_wrong_to_wrong": round(n_flip_w2w / denom, 4),
            "net_correction_benefit": round((n_flip_w2c - n_flip_c2w) / denom, 4),
            "n_rollout_steps": n_steps,
        }

    # ── 4. Rollout Diversity ─────────────────────────────────────────────────
    n_unique_list, prm_stds, prm_ranges = [], [], []
    n_diverse = 0
    depth_agreement: Dict[int, int] = {d: 0 for d in range(1, D + 1)}

    for q in questions:
        rollouts = [r for r in (q.get("rollouts") or []) if r]
        final_ans_norms = [_norm(str(r.get("final_answer") or "")) for r in rollouts]
        non_empty = [a for a in final_ans_norms if a]
        n_u = len(set(non_empty))
        n_unique_list.append(n_u)
        if n_u > 1:
            n_diverse += 1

        final_prms = [float(r.get("final_prm_score") or 0.0) for r in rollouts]
        if len(final_prms) >= 2:
            prm_stds.append(_safe_std(final_prms))
            prm_ranges.append(max(final_prms) - min(final_prms))

        for d in range(1, D + 1):
            d_answers = []
            for r in rollouts:
                s = _step_at_depth(r, d)
                if s:
                    a = _norm(str(s.get("corrected_answer") or s.get("reasoning_answer") or ""))
                    if a:
                        d_answers.append(a)
            if d_answers and len(set(d_answers)) == 1:
                depth_agreement[d] += 1

    diversity = {
        "mean_n_unique_final_answers": round(_safe_mean(n_unique_list), 4),
        "frac_questions_diverse": round(n_diverse / denom_total, 4),
        "mean_final_prm_std": round(_safe_mean(prm_stds), 4),
        "mean_final_prm_range": round(_safe_mean(prm_ranges), 4),
        "depth_wise_answer_agreement": {
            str(d): round(depth_agreement[d] / denom_total, 4)
            for d in range(1, D + 1)
        },
    }

    # ── 5. PRM Calibration ───────────────────────────────────────────────────
    prm_correct, prm_incorrect = [], []
    prm_maj_correct, prm_maj_incorrect = [], []
    for q in questions:
        agg = q.get("aggregation") or {}
        max_prm_score = float(agg.get("max_prm_score") or 0.0)
        maj_correct = agg.get("majority_is_correct")
        max_prm_correct_flag = agg.get("max_prm_is_correct")

        if max_prm_correct_flag is True:
            prm_correct.append(max_prm_score)
        elif max_prm_correct_flag is False:
            prm_incorrect.append(max_prm_score)

        # PRM of the majority-selected rollout
        maj_idx = int(agg.get("majority_rollout_idx") or -1)
        rollouts = q.get("rollouts") or []
        for r in rollouts:
            if r and int(r.get("rollout_idx") or -1) == maj_idx:
                maj_prm = float(r.get("final_prm_score") or 0.0)
                if maj_correct is True:
                    prm_maj_correct.append(maj_prm)
                elif maj_correct is False:
                    prm_maj_incorrect.append(maj_prm)

    mean_prm_corr = _safe_mean(prm_correct)
    mean_prm_incorr = _safe_mean(prm_incorrect)
    prm_calibration = {
        "mean_max_prm_score_correct": round(mean_prm_corr, 4),
        "mean_max_prm_score_incorrect": round(mean_prm_incorr, 4),
        "prm_score_gap": round(mean_prm_corr - mean_prm_incorr, 4),
        "majority_correct_mean_prm": round(_safe_mean(prm_maj_correct), 4),
        "majority_incorrect_mean_prm": round(_safe_mean(prm_maj_incorrect), 4),
        "n_evaluated_correct": len(prm_correct),
        "n_evaluated_incorrect": len(prm_incorrect),
    }

    # ── 6. Voting Statistics ─────────────────────────────────────────────────
    vote_margins = []
    n_agree = n_disagree = 0
    n_dis_maj_correct = n_dis_prm_correct = 0

    for q in questions:
        agg = q.get("aggregation") or {}
        n_r = int(agg.get("n_rollouts") or N or 1)
        vc = int(agg.get("majority_vote_count") or 0)
        if vc > 0 and n_r > 0:
            vote_margins.append(vc / n_r)

        maj_norm = _norm(agg.get("majority_pred") or "")
        prm_norm = _norm(agg.get("max_prm_pred") or "")
        if maj_norm and prm_norm:
            if maj_norm == prm_norm:
                n_agree += 1
            else:
                n_disagree += 1
                if agg.get("majority_is_correct") is True:
                    n_dis_maj_correct += 1
                if agg.get("max_prm_is_correct") is True:
                    n_dis_prm_correct += 1

    n_both = max(n_agree + n_disagree, 1)
    voting = {
        "mean_majority_vote_margin": round(_safe_mean(vote_margins), 4),
        "majority_maxprm_agree_rate": round(n_agree / n_both, 4),
        "majority_correct_when_disagree": round(n_dis_maj_correct / max(n_disagree, 1), 4),
        "max_prm_correct_when_disagree": round(n_dis_prm_correct / max(n_disagree, 1), 4),
        "n_disagreements": n_disagree,
    }

    # ── 7. PRM Monotonicity across depths ────────────────────────────────────
    n_monotone = n_ever_improve = 0
    final_vs_initial_lifts = []
    n_rollouts_analyzed = 0

    for q in questions:
        for r in (q.get("rollouts") or []):
            if not r:
                continue
            prms_per_depth = []
            for d in range(1, D + 1):
                s = _step_at_depth(r, d)
                if s:
                    prms_per_depth.append(float(s.get("corrected_prm_score") or 0.0))

            if len(prms_per_depth) < 2:
                continue
            n_rollouts_analyzed += 1

            if all(prms_per_depth[i] <= prms_per_depth[i + 1]
                   for i in range(len(prms_per_depth) - 1)):
                n_monotone += 1

            if any(prms_per_depth[i + 1] > prms_per_depth[i]
                   for i in range(len(prms_per_depth) - 1)):
                n_ever_improve += 1

            final_vs_initial_lifts.append(prms_per_depth[-1] - prms_per_depth[0])

    denom_rollouts = max(n_rollouts_analyzed, 1)
    prm_monotonicity = {
        "frac_rollouts_monotone_up": round(n_monotone / denom_rollouts, 4),
        "frac_rollouts_ever_improve": round(n_ever_improve / denom_rollouts, 4),
        "mean_final_vs_initial_prm_lift": round(_safe_mean(final_vs_initial_lifts), 4),
        "n_rollouts_analyzed": n_rollouts_analyzed,
    }

    # ── 8. Efficiency ────────────────────────────────────────────────────────
    q_elapsed = [float(q.get("elapsed_sec") or 0.0) for q in questions]
    r_elapsed = [
        float(r.get("elapsed_sec") or 0.0)
        for q in questions for r in (q.get("rollouts") or []) if r
    ]
    efficiency = {
        "mean_elapsed_sec_per_question": round(_safe_mean(q_elapsed), 3),
        "total_elapsed_sec": round(sum(q_elapsed), 3),
        "mean_elapsed_sec_per_rollout": round(_safe_mean(r_elapsed), 3),
    }

    return {
        "dataset": _infer_dataset(source_file),
        "model": _infer_model(source_file, source_file),
        "N": N,
        "D": D,
        "n_questions": n_total,
        "n_questions_with_gold": n_with_gold,
        "accuracy": accuracy,
        "prm_dynamics": prm_dynamics,
        "correction_effectiveness": correction_eff,
        "diversity": diversity,
        "prm_calibration": prm_calibration,
        "voting": voting,
        "prm_monotonicity": prm_monotonicity,
        "efficiency": efficiency,
    }


# ─── Path helpers ─────────────────────────────────────────────────────────────

def _infer_dataset(path: str) -> str:
    m = re.search(r"cimr_full_results_(.+?)_N\d", os.path.basename(path))
    return m.group(1) if m else ""


def _infer_model(path: str, fallback: str) -> str:
    parts = path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part.startswith("cimr_full_results_") and i > 0:
            return parts[i - 1]
    return fallback


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_store(input_path: str) -> Optional[dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(r.get("id", i)): r for i, r in enumerate(data)}
    if isinstance(data, dict):
        return data
    logger.warning(f"Unexpected top-level type in {input_path}; skipping.")
    return None


def _write_summary(summary: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, out_path)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract refinement-dynamics metrics from result stores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_dir",
        default="experiments_results/results_paired",
        help="Root folder containing cimr_full_results_*.json files (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default="experiments_results/cimr_metrics",
        help="Root folder for output metrics_summary.json files (default: %(default)s)",
    )
    args = parser.parse_args()

    # Group input files by their immediate parent directory (one summary per model folder).
    # dir_rel → { filename → metrics_dict }
    per_dir: Dict[str, Dict[str, Any]] = {}

    n_ok = n_err = 0
    for root, dirs, files in os.walk(args.results_dir):
        dirs.sort()
        for fname in sorted(files):
            if not (fname.startswith("cimr_full_results_") and fname.endswith(".json")):
                continue
            full_path = os.path.join(root, fname)
            try:
                store = _load_store(full_path)
                if store is None:
                    continue
                metrics = extract_metrics(store, full_path)
                dir_rel = os.path.relpath(root, args.results_dir)
                per_dir.setdefault(dir_rel, {})[fname] = metrics
                logger.info(f"  processed  {os.path.relpath(full_path)}")
                n_ok += 1
            except Exception as e:
                logger.error(f"Failed on {full_path}: {e}")
                n_err += 1

    # Write one metrics_summary.json per model directory.
    for dir_rel, summary in per_dir.items():
        out_path = os.path.join(args.output_dir, dir_rel, "metrics_summary.json")
        _write_summary(summary, out_path)
        logger.info(f"  written  {os.path.relpath(out_path)}")

    logger.info(f"Done. {n_ok} files processed into {len(per_dir)} summary file(s), {n_err} errors.")


if __name__ == "__main__":
    main()
