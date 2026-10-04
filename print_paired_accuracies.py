#!/usr/bin/env python3
"""
Print a table of breadth-depth refinement accuracies across runs.

Reads ``summary_paired_<dataset>_N<N>_d<D>[_no_critique][_no_correction].json``
files under ``--results_dir`` (written by ``run_paired_ablation.py``). If none
are found, accuracies are recomputed from the ``cimr_full_results_*.json``
stores instead. ``cimr_majority`` is the method's prediction; ``cimr_max_prm``
is the diagnostic max-PRM selection.

Usage
-----
    python print_paired_accuracies.py \
        [--results_dir experiments_results/results_paired] \
        [--dataset amc23] [--method cimr_majority] [--out_csv accuracies.csv]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_TAG = r"(?P<dataset>.+?)_N(?P<N>\d+)_d(?P<D>\d+)(?P<suffix>(?:_no_critique|_no_correction)*)"
_SUMMARY_RE = re.compile(rf"^summary_paired_{_TAG}\.json$")
_CIMR_FULL_RE = re.compile(rf"^cimr_full_results_{_TAG}\.json$")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def collect_rows(
    results_dir: Path,
    dataset_filter: Optional[str] = None,
    method_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for path in sorted(results_dir.rglob("summary_paired_*.json")):
        m = _SUMMARY_RE.match(path.name)
        if not m:
            continue

        dataset = m.group("dataset")
        n_value = _safe_int(m.group("N"), 0)
        d_value = _safe_int(m.group("D"), 0)
        ablation = m.group("suffix").lstrip("_") or "none"

        if dataset_filter and dataset != dataset_filter:
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] Could not parse {path}: {e}", file=sys.stderr)
            continue

        if not isinstance(payload, dict):
            continue

        # The immediate parent folder is usually the model run folder.
        model_run = path.parent.name

        for method, data in payload.items():
            if method.startswith("__"):
                continue
            if method_filter and method != method_filter:
                continue
            if not isinstance(data, dict):
                continue

            n_correct = _safe_int(data.get("n_correct"), 0)
            n_total = _safe_int(data.get("n_total"), 0)

            # Prefer explicit accuracy; otherwise derive from counts when possible.
            if "accuracy" in data:
                accuracy = _safe_float(data.get("accuracy"), float("nan"))
            elif n_total > 0:
                accuracy = 100.0 * n_correct / n_total
            else:
                # Keep row visible even if accuracy is unavailable.
                accuracy = float("nan")

            # If there's no useful accuracy/count signal at all, skip.
            if ("accuracy" not in data) and n_total <= 0:
                continue

            rows.append(
                {
                    "model_run": model_run,
                    "dataset": dataset,
                    "N": n_value,
                    "D": d_value,
                    "ablation": ablation,
                    "method": method,
                    "accuracy": accuracy,
                    "n_correct": n_correct,
                    "n_total": n_total,
                    "summary_file": str(path),
                }
            )

    rows.sort(key=lambda r: (r["model_run"], r["dataset"], r["N"], r["D"], r["ablation"], r["method"]))
    return rows


def _collect_rows_from_cimr_full(
    results_dir: Path,
    dataset_filter: Optional[str] = None,
    method_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fallback: recompute accuracies from the cimr_full_results stores."""
    rows: List[Dict[str, Any]] = []

    for path in sorted(results_dir.rglob("cimr_full_results_*.json")):
        m = _CIMR_FULL_RE.match(path.name)
        if not m:
            continue

        dataset = m.group("dataset")
        n_value = _safe_int(m.group("N"), 0)
        d_value = _safe_int(m.group("D"), 0)
        ablation = m.group("suffix").lstrip("_") or "none"
        if dataset_filter and dataset != dataset_filter:
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] Could not parse {path}: {e}", file=sys.stderr)
            continue
        if not isinstance(payload, dict):
            continue

        model_run = path.parent.name

        n_questions = 0
        maj_scored = 0
        prm_scored = 0
        maj_correct = 0
        prm_correct = 0
        for _, qrec in payload.items():
            if not isinstance(qrec, dict):
                continue
            agg = qrec.get("aggregation")
            if not isinstance(agg, dict):
                continue
            n_questions += 1
            maj = agg.get("majority_is_correct")
            prm = agg.get("max_prm_is_correct")
            if maj is not None:
                maj_scored += 1
                maj_correct += int(bool(maj))
            if prm is not None:
                prm_scored += 1
                prm_correct += int(bool(prm))

        if n_questions <= 0:
            continue

        maj_acc = (100.0 * maj_correct / maj_scored) if maj_scored > 0 else float("nan")
        prm_acc = (100.0 * prm_correct / prm_scored) if prm_scored > 0 else float("nan")

        maybe_rows = [
            {
                "model_run": model_run,
                "dataset": dataset,
                "N": n_value,
                "D": d_value,
                "ablation": ablation,
                "method": "cimr_majority",
                "accuracy": maj_acc,
                "n_correct": maj_correct,
                "n_total": maj_scored,
                "summary_file": str(path),
            },
            {
                "model_run": model_run,
                "dataset": dataset,
                "N": n_value,
                "D": d_value,
                "ablation": ablation,
                "method": "cimr_max_prm",
                "accuracy": prm_acc,
                "n_correct": prm_correct,
                "n_total": prm_scored,
                "summary_file": str(path),
            },
        ]

        for r in maybe_rows:
            if method_filter and r["method"] != method_filter:
                continue
            rows.append(r)

    rows.sort(key=lambda r: (r["model_run"], r["dataset"], r["N"], r["D"], r["ablation"], r["method"]))
    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No paired summary accuracies found.")
        return

    headers = ["model_run", "dataset", "N", "D", "ablation", "method", "accuracy", "correct/total"]

    formatted: List[List[str]] = []
    for r in rows:
        formatted.append(
            [
                str(r["model_run"]),
                str(r["dataset"]),
                str(r["N"]),
                str(r["D"]),
                str(r["ablation"]),
                str(r["method"]),
                (f"{r['accuracy']:.2f}%" if not math.isnan(float(r["accuracy"])) else "N/A"),
                f"{r['n_correct']}/{r['n_total']}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in formatted:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(parts: List[str]) -> str:
        return " | ".join(parts[i].ljust(widths[i]) for i in range(len(parts)))

    sep = "-+-".join("-" * w for w in widths)

    print(_line(headers))
    print(sep)
    for row in formatted:
        print(_line(row))


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model_run",
                "dataset",
                "N",
                "D",
                "ablation",
                "method",
                "accuracy",
                "n_correct",
                "n_total",
                "summary_file",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"CSV saved: {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print breadth-depth refinement accuracies across runs."
    )
    p.add_argument(
        "--results_dir",
        default="./experiments_results/results_paired",
        help="Root paired-results folder.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print discovered files and fallback decisions.",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset filter (e.g., aime24).",
    )
    p.add_argument(
        "--method",
        default=None,
        help="Optional method filter (e.g., cimr_majority).",
    )
    p.add_argument(
        "--out_csv",
        default=None,
        help="Optional CSV output path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    summary_files = sorted(results_dir.rglob("summary_paired_*.json"))
    cimr_full_files = sorted(results_dir.rglob("cimr_full_results_*.json"))
    if args.debug:
        print(f"Scanning results_dir: {results_dir.resolve()}")
        print(f"  summary_paired files found: {len(summary_files)}")
        print(f"  cimr_full files found: {len(cimr_full_files)}")

    rows = collect_rows(
        results_dir=results_dir,
        dataset_filter=args.dataset,
        method_filter=args.method,
    )

    if not rows:
        # No summary files: fall back to the full result stores.
        rows = _collect_rows_from_cimr_full(
            results_dir=results_dir,
            dataset_filter=args.dataset,
            method_filter=args.method,
        )
        if args.debug and rows:
            print("Using fallback parser from cimr_full_results_*.json")

    print_table(rows)

    if args.out_csv:
        write_csv(rows, Path(args.out_csv))


if __name__ == "__main__":
    main()
