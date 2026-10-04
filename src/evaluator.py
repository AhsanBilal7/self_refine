"""Resumable evaluation loop for the baselines in ``inference_methods``.

* Every question is appended to the JSONL file and flushed as soon as it is
  scored, and the summary JSON is rewritten atomically after each question.
* On restart, questions already present in the JSONL (matched on the first
  200 characters of the question) are skipped and the summary is rebuilt from
  the saved rows.
* Greedy decoding batches ``model.batch_size`` questions per ``generate()``
  call; sampling methods decode a flat list of ``Q x N`` prompts and score each
  question as soon as its ``N`` samples are complete.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from tqdm import tqdm

from .inference_methods import (
    BestOfN,
    BestOfNMajority,
    GreedyInference,
    GreedyPassN,
    InferenceResult,
    _check_correct,
    _extract_pred,
    _make_error_result,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Atomic summary writer
# ──────────────────────────────────────────────────────────────────────────────

def _write_summary_atomic(summary: Dict[str, Any], path: str) -> None:
    """
    Write summary JSON atomically via a temp file + os.replace().
    Guarantees the summary file is never partially written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, path)  # atomic on POSIX systems


# ──────────────────────────────────────────────────────────────────────────────
# JSONL reader  (used for resume)
# ──────────────────────────────────────────────────────────────────────────────

def _load_completed(jsonl_path: str):
    """
    Read an existing JSONL file and return:
      done_keys  : set of question[:200] strings already completed
      rows       : list of all completed row dicts
      n_correct  : count of correct answers in completed rows
      n_total    : count of rows with is_correct not None
      total_time : sum of elapsed_sec across completed rows
    """
    done_keys  = set()
    rows       = []
    n_correct  = 0
    n_total    = 0
    total_time = 0.0

    if not os.path.isfile(jsonl_path):
        return done_keys, rows, n_correct, n_total, total_time

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = r.get("question", "")[:200]
                if key in done_keys:
                    continue  # skip duplicate lines from interrupted writes
                done_keys.add(key)
                rows.append(r)
                if r.get("is_correct") is not None:
                    n_correct += int(r["is_correct"])
                    n_total   += 1
                total_time += r.get("elapsed_sec", 0.0)
            except Exception:
                pass  # skip malformed lines

    return done_keys, rows, n_correct, n_total, total_time


# ──────────────────────────────────────────────────────────────────────────────
# Per-question result generators
# ──────────────────────────────────────────────────────────────────────────────

def _greedy_results(method, pending, max_new_tokens) -> Iterator[tuple]:
    """Yield ``(example, InferenceResult)`` for greedy decoding, chunk by chunk."""
    bs = method.model.batch_size
    n  = len(pending)

    chunk_bar = tqdm(
        range(0, n, bs),
        total=max(1, -(-n // bs)),  # ceil division
        desc=f"[Greedy] generate (bs={bs})",
        unit="chunk",
        dynamic_ncols=True,
        colour="green",
    )

    for start in chunk_bar:
        chunk_ex  = pending[start : start + bs]
        questions = [ex["question"] for ex in chunk_ex]
        golds     = [ex.get("answer") for ex in chunk_ex]

        t0     = time.time()
        traces = method.model.generate_greedy_batch(
            questions, max_new_tokens=max_new_tokens
        )
        elapsed = time.time() - t0
        per_q   = round(elapsed / max(len(chunk_ex), 1), 4)

        chunk_bar.set_postfix(
            questions=f"{min(start + bs, n)}/{n}", refresh=False
        )

        for ex, gold, trace in zip(chunk_ex, golds, traces):
            pred    = _extract_pred(trace, method.ev, method.data_name)
            correct = _check_correct(pred, gold, method.ev, method.data_name)
            res = InferenceResult(
                method="greedy", question=ex["question"],
                gold_answer=gold, selected_trace=trace,
                predicted_answer=pred, is_correct=correct,
                N=1, elapsed_sec=per_q,
                all_traces=[trace], selected_index=0,
            )
            yield ex, res

    chunk_bar.close()


def _sampling_results(method, pending, max_new_tokens, label, colour) -> Iterator[tuple]:
    """Yield ``(example, InferenceResult)`` for a sampling method.

    Consumes ``model.generate_samples_flat`` and applies ``method._select`` as
    soon as a question's ``N`` samples are available. If flat generation
    fails, the remaining questions are generated one at a time.
    """
    bs = method.model.batch_size
    N  = method.N
    questions = [ex["question"] for ex in pending]
    golds     = [ex.get("answer") for ex in pending]

    logger.info(
        f"  Flat-batch generation: Q={len(questions)}  N={N}  "
        f"batch_size={bs}  →  {len(questions)*N} total seqs  "
        f"→  {-(-len(questions)*N // bs)} generate() calls  "
        f"→  saving every {bs // N} questions"
    )

    n_correct = 0
    n_done    = 0

    save_bar = tqdm(
        total=len(pending),
        desc=f"{label}",
        unit="q",
        dynamic_ncols=True,
        colour=colour,
    )

    try:
        for q_idx, traces in method.model.generate_samples_flat(
            questions, N=N,
            temperature=method.temperature, top_p=method.top_p,
            max_new_tokens=max_new_tokens,
        ):
            ex   = pending[q_idx]
            gold = golds[q_idx]

            try:
                res = method._select(ex["question"], gold, traces, elapsed=0.0)
            except Exception as e:
                logger.error(f"  _select q[{q_idx}] failed: {e}")
                res = _make_error_result(
                    method.__class__.__name__.lower(), ex["question"], gold, N
                )

            n_done += 1
            if res.is_correct is not None:
                n_correct += int(res.is_correct)
                save_bar.set_postfix(
                    acc=f"{n_correct/n_done:.3f}",
                    correct=f"{n_correct}/{n_done}",
                    refresh=False,
                )
            save_bar.update(1)
            yield ex, res

    except Exception as e:
        logger.error(f"  generate_samples_flat failed: {e}. Falling back to per-question.")
        # Fallback: per-question loop
        for i, (ex, gold) in enumerate(zip(pending[n_done:], golds[n_done:]), start=n_done):
            try:
                traces = method.model.generate_samples(
                    ex["question"], N=N,
                    temperature=method.temperature, top_p=method.top_p,
                    max_new_tokens=max_new_tokens,
                )
                res = method._select(ex["question"], gold, traces, elapsed=0.0)
            except Exception as e2:
                logger.error(f"  fallback q[{i}] failed: {e2}")
                res = _make_error_result(
                    method.__class__.__name__.lower(), ex["question"], gold, N
                )
            n_done += 1
            save_bar.update(1)
            yield ex, res
    finally:
        save_bar.close()


# ──────────────────────────────────────────────────────────────────────────────
# Primary evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_batch(
    method,
    examples: List[Dict[str, Any]],
    max_new_tokens: int = 1024,
    save_path: Optional[str] = None,
    summary_path: Optional[str] = None,
    resume: bool = True,
    log_every: int = 10,
    summary_every: int = 1,    # rewrite summary every N questions (default: every question)
) -> Dict[str, Any]:
    """
    Run inference over `examples`, saving every result immediately.

    Parameters
    ----------
    method         : GreedyInference / BestOfN / BestOfNMajority / GreedyPassN
    examples       : list of {'question', 'answer', '_data_name', ...}
    max_new_tokens : generation length cap
    save_path      : JSONL file — one line appended per question, immediately
    summary_path   : summary JSON — atomically rewritten every `summary_every`
                     questions so it always reflects current progress
    resume         : skip already-done questions by reading existing JSONL
    log_every      : structured logger interval
    summary_every  : how often to rewrite the summary JSON (default 1 = always)

    Returns
    -------
    dict: accuracy, n_correct, n_total, avg_elapsed_sec, total_wall_sec, results
    """
    method_name = method.__class__.__name__

    # ── load already-completed results from previous run ──────────────────────
    done_keys  = set()
    rows       = []
    n_correct  = 0
    n_total    = 0
    total_time = 0.0

    if resume and save_path:
        done_keys, rows, n_correct, n_total, total_time = _load_completed(save_path)
        if done_keys:
            logger.info(
                f"Resuming: {len(done_keys)} questions already done  "
                f"(acc so far: {n_correct}/{n_total})"
            )

    pending = [ex for ex in examples
               if ex["question"][:200] not in done_keys]

    logger.info(
        f"Evaluating {len(pending)}/{len(examples)} pending questions  "
        f"(method={method_name})"
    )

    if not pending:
        logger.info("All questions already completed — nothing to do.")
        return _build_return(rows, n_correct, n_total, total_time)

    # ── open JSONL for appending ───────────────────────────────────────────────
    f_out = None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        f_out = open(save_path, "a", encoding="utf-8")

    # ── choose the result generator ───────────────────────────────────────────
    if isinstance(method, GreedyInference):
        result_gen = _greedy_results(method, pending, max_new_tokens)
    elif isinstance(method, BestOfN):
        result_gen = _sampling_results(
            method, pending, max_new_tokens,
            label=f"[BoN N={method.N}]", colour="blue",
        )
    elif isinstance(method, GreedyPassN):
        result_gen = _sampling_results(
            method, pending, max_new_tokens,
            label=f"[GreedyPass@N N={method.N}]", colour="cyan",
        )
    elif isinstance(method, BestOfNMajority):
        result_gen = _sampling_results(
            method, pending, max_new_tokens,
            label=f"[BoN-Majority N={method.N}]", colour="yellow",
        )
    else:
        raise TypeError(f"Unknown method type: {type(method)}")

    # ── main loop: generate → save JSONL → update summary ────────────────────
    t_wall_start = time.time()
    n_done_this_run = 0

    try:
        for ex, res in result_gen:
            row = res.to_dict()
            row["question"] = ex["question"][:300]
            rows.append(row)

            # Update running counts
            if res.is_correct is not None:
                n_correct += int(res.is_correct)
                n_total   += 1
            total_time += res.elapsed_sec
            n_done_this_run += 1

            if f_out:
                f_out.write(json.dumps(row) + "\n")
                f_out.flush()


            if summary_path and (n_done_this_run % summary_every == 0):
                summary = _build_return(rows, n_correct, n_total, total_time)
                _write_summary_atomic(summary, summary_path)

            total_done = len(done_keys) + n_done_this_run
            if total_done % log_every == 0:
                acc = n_correct / max(n_total, 1)
                logger.info(
                    f"  [{total_done}/{len(examples)}] "
                    f"acc={acc:.3f} ({n_correct}/{n_total})"
                )
    finally:
        if f_out:
            f_out.close()
        # Write the summary even when the loop was interrupted.
        if summary_path and rows:
            summary = _build_return(rows, n_correct, n_total, total_time)
            _write_summary_atomic(summary, summary_path)
            logger.info(f"Summary (final) saved → {summary_path}")

    total_wall = time.time() - t_wall_start
    result = _build_return(rows, n_correct, n_total, total_time)
    result["total_wall_sec"] = round(total_wall, 2)

    logger.info(
        f"Done. acc={result['accuracy']:.4f} ({n_correct}/{n_total})  "
        f"wall={total_wall:.1f}s  avg={result['avg_elapsed_sec']:.2f}s/q"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Summary builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_return(
    rows: List[Dict],
    n_correct: int,
    n_total: int,
    total_time: float,
) -> Dict[str, Any]:
    accuracy    = n_correct / max(n_total, 1)
    avg_elapsed = total_time / max(len(rows), 1)
    return {
        "accuracy":        round(accuracy, 4),
        "n_correct":       n_correct,
        "n_total":         n_total,
        "avg_elapsed_sec": round(avg_elapsed, 3),
        "total_wall_sec":  0.0,   # filled in by caller
        "results":         rows,
    }


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_summary(summary: Dict[str, Any], path: str):
    """Write summary JSON atomically."""
    _write_summary_atomic(summary, path)
    logger.info(f"Summary saved to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def compare_methods(summaries: Dict[str, Dict[str, Any]]) -> str:
    """Plain-text accuracy / timing table across methods."""
    lines = [
        f"\n{'Method':<20} {'Accuracy':>10} {'Correct':>10} "
        f"{'Total':>8} {'Avg time':>10} {'Wall time':>12}",
        "-" * 74,
    ]
    for name, s in summaries.items():
        wall = s.get("total_wall_sec", 0.0)
        lines.append(
            f"{name:<20} {s['accuracy']:>10.4f} {s['n_correct']:>10} "
            f"{s['n_total']:>8} {s['avg_elapsed_sec']:>9.2f}s"
            f" {wall:>11.1f}s"
        )
    return "\n".join(lines)


def redundancy_stats(results: List[Dict]) -> Dict[str, float]:
    """Vote-group statistics for ``bon_majority`` rows.

    Averages the number of distinct answer groups, the size of the selected
    group and of the largest group, and how often the selected group is the
    largest one.
    """
    n_cl, sel_sz, max_sz, top_sel = [], [], [], []
    for r in results:
        sizes = r.get("cluster_sizes", [])
        if not sizes:
            continue
        size_arr       = np.array(sizes)
        unique_clusters = len(set(r.get("cluster_labels", range(len(sizes)))))
        n_cl.append(unique_clusters)
        sel_idx = r.get("selected_index", 0)
        sel_sz.append(sizes[sel_idx] if sel_idx < len(sizes) else 0)
        max_sz.append(int(np.max(size_arr)))
        top_sel.append(1 if sizes[sel_idx] == np.max(size_arr) else 0)
    if not n_cl:
        return {}
    return {
        "avg_num_clusters":               round(float(np.mean(n_cl)), 3),
        "avg_selected_cluster_size":      round(float(np.mean(sel_sz)), 3),
        "avg_max_cluster_size":           round(float(np.mean(max_sz)), 3),
        "frac_selected_from_top_cluster": round(float(np.mean(top_sel)), 3),
    }