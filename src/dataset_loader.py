"""Benchmark loading with clean gold answers.

Every example returned by ``load_dataset_split`` has

    question   : problem text
    answer     : final gold answer only (e.g. "2", "\\frac{1}{3}"), never a
                 full worked solution
    _data_name : canonical dataset name
    _raw       : the original Hugging Face row

Gold-answer extraction per dataset:

    math500        last \\boxed{} of the ``solution`` field
    aime24         ``Answer``
    aime25, amc23  ``answer``
    olympiadbench  ``final_answer[0]`` with surrounding ``$`` removed
    gsm8k          text after ``####`` in ``answer``
    custom         ``answer`` (or ``solution``; a \\boxed{} is unwrapped)
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Hugging Face dataset registry
# ──────────────────────────────────────────────────────────────────────────────

_HF_CONFIGS: Dict[str, Dict[str, Any]] = {
    "math500": {
        "hf_path":      "HuggingFaceH4/MATH-500",
        "hf_name":      "default",
        "split":        "test",
        "question_key": "problem",
        "answer_key":   "solution",
    },
    "aime24": {
        "hf_path":      "Maxwell-Jia/AIME_2024",
        "hf_name":      None,
        "split":        "train",
        "question_key": "Problem",
        "answer_key":   "Answer",
    },
    "aime25": {
        "hf_path":      "math-ai/aime25",
        "hf_name":      None,
        "split":        "test",
        "question_key": "problem",
        "answer_key":   "answer",
    },
    "amc23": {
        "hf_path":      "math-ai/amc23",
        "hf_name":      None,
        "split":        "test",
        "question_key": "question",
        "answer_key":   "answer",
    },
    "olympiadbench": {
        "hf_path":      "math-ai/olympiadbench",
        "hf_name":      None,
        "split":        "test",
        "question_key": "question",
        "answer_key":   "final_answer",
    },
    "gsm8k": {
        "hf_path":      "openai/gsm8k",
        "hf_name":      "main",
        "split":        "test",
        "question_key": "question",
        "answer_key":   "answer",
    },
}

DATASETS = sorted(_HF_CONFIGS) + ["custom"]


# ──────────────────────────────────────────────────────────────────────────────
# Gold-answer extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_gt(row: Dict, data_name: str) -> str:
    """Return the clean final-answer string for one dataset row."""
    answer_key = _HF_CONFIGS.get(data_name, {}).get("answer_key", "answer")

    if data_name in ("math500", "math"):
        return _parse_boxed(str(row.get(answer_key, "")))

    if data_name in ("aime24", "aime25", "amc23"):
        return str(row.get(answer_key, "")).strip()

    if data_name == "gsm8k":
        raw = str(row.get(answer_key, ""))
        return raw.split("####")[-1].strip() if "####" in raw else raw.strip()

    if data_name == "olympiadbench":
        fa = row.get(answer_key, [""])
        val = fa[0] if isinstance(fa, list) and fa else str(fa)
        return str(val).strip("$").strip()

    # Fallback: first available answer-like field.
    val = row.get(answer_key)
    if val is None:
        for key in ("answer", "Answer", "solution", "target", "label"):
            val = row.get(key)
            if val is not None:
                break
    if val is None:
        return ""
    raw = str(val).strip()
    if "####" in raw:
        return raw.split("####")[-1].strip()
    if "boxed" in raw:
        return _parse_boxed(raw)
    return raw.strip("$").strip()


def _parse_boxed(text: str) -> str:
    """Content of the last ``\\boxed{...}`` (brace-matched), or ''."""
    if "boxed" not in text:
        return ""
    ans = text.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack, result = 1, ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                result += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                result += c
            else:
                result += c
        return result.strip()
    # \boxed without braces: take everything up to the next $.
    return ans.split("$")[0].strip()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset_split(
    data_name: str,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    custom_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load a benchmark and return normalised examples.

    Parameters
    ----------
    data_name   : one of ``DATASETS``.
    split       : override the default split.
    max_samples : keep only the first ``max_samples`` examples.
    custom_path : local JSON/JSONL file (required for ``custom``).
    cache_dir   : Hugging Face cache directory.
    """
    if data_name == "custom":
        return _load_custom(custom_path, max_samples)

    if data_name not in _HF_CONFIGS:
        raise ValueError(f"Unknown dataset '{data_name}'. Choose from: {DATASETS}.")

    cfg = _HF_CONFIGS[data_name]
    effective_split = split or cfg["split"]

    try:
        from datasets import load_dataset as hf_load
        kwargs: Dict[str, Any] = {"split": effective_split}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if cfg.get("hf_name"):
            raw_ds = hf_load(cfg["hf_path"], cfg["hf_name"], **kwargs)
        else:
            raw_ds = hf_load(cfg["hf_path"], **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load '{data_name}' from the Hugging Face Hub: {e}"
        ) from e

    examples: List[Dict[str, Any]] = []
    n_empty = 0
    for row in raw_ds:
        row_dict = dict(row)
        question = str(row_dict.get(cfg["question_key"], "")).strip()
        if not question:
            continue

        answer = _extract_gt(row_dict, data_name)
        if not answer:
            n_empty += 1
            logger.debug(f"Empty answer for row: {row_dict}")

        examples.append({
            "question":   question,
            "answer":     answer,
            "_data_name": data_name,
            "_raw":       row_dict,
        })
        if max_samples and len(examples) >= max_samples:
            break

    if n_empty:
        logger.warning(
            f"{n_empty}/{len(examples)} examples had empty answers; "
            f"check the dataset fields."
        )
    logger.info(
        f"Loaded {len(examples)} examples from '{data_name}' "
        f"({effective_split} split).  "
        f"Sample answers: {[e['answer'] for e in examples[:3]]}"
    )
    return examples


def _load_custom(path: Optional[str], max_samples: Optional[int]) -> List[Dict]:
    """Load ``question``/``problem`` + ``answer``/``solution`` rows from JSON(L)."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Custom file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = json.load(f)
            if isinstance(rows, dict):
                rows = list(rows.values())

    examples = []
    for row in rows:
        question = str(row.get("question", row.get("problem", ""))).strip()
        raw_ans = str(row.get("answer", row.get("solution", ""))).strip()
        answer = _parse_boxed(raw_ans) if "boxed" in raw_ans else raw_ans
        if question:
            examples.append({
                "question":   question,
                "answer":     answer,
                "_data_name": "custom",
                "_raw":       row,
            })
        if max_samples and len(examples) >= max_samples:
            break

    logger.info(f"Loaded {len(examples)} custom examples from {path}.")
    return examples


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Print a few examples of a benchmark.")
    ap.add_argument("--dataset", default="math500", choices=DATASETS)
    ap.add_argument("--max_samples", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    for i, ex in enumerate(load_dataset_split(args.dataset, max_samples=args.max_samples)):
        print(f"\n[{i}] Q: {ex['question'][:100]}...")
        print(f"    A: {ex['answer']}")
