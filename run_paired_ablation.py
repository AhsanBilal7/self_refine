"""Run breadth-depth refinement on one benchmark.

For each problem: sample N rollouts, refine each for D rounds of reasoning
continuation -> self-critique -> self-correction, and take a plurality vote
over the terminal answers (paper Algorithm 1). Also reports the diagnostic
max-PRM selection over the same rollouts.

Ablations
---------
  --no_critique    the corrector receives an empty critique (w/o-critic row)
  --no_correction  the corrector is skipped; r^(d) = g^(d)
  --no_prm         do not load the PRM; diagnostic scores are 0 and vote ties
                   fall back to rollout order

Outputs (in --output_dir), tagged <dataset>_N<N>_d<D>[_no_critique][_no_correction]
--------------------------------------------------------------------------------
  cimr_full_results_<tag>.json   full per-question / rollout / step store
                                 (resumable; the input to the analysis scripts)
  summary_paired_<tag>.json      accuracy of cimr_majority and cimr_max_prm
  results_paired_<tag>.jsonl     one prediction record per question

Example
-------
  python run_paired_ablation.py --model Qwen/Qwen2.5-Math-7B-Instruct \\
      --dataset amc23 --cimr_n_rollouts 8 --cimr_depth 4 --output_dir out/
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

from src.dataset_loader import DATASETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paired_ablation")

SELECTION_RULES = ("majority", "max_prm")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Model")
    g.add_argument("--model", required=True, help="Hugging Face model id or local path")
    g.add_argument("--batch_size", type=int, default=8,
                   help="Sequences per generate() call")
    g.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    g.add_argument("--device", default="auto")
    g.add_argument("--gpus", default=None, help="Sets CUDA_VISIBLE_DEVICES, e.g. '0,1'")
    g.add_argument("--load_in_4bit", action="store_true")
    g.add_argument("--load_in_8bit", action="store_true")
    g.add_argument("--trust_remote_code", action="store_true")
    g.add_argument("--hf_cache_dir", default=None)

    g = p.add_argument_group("Diagnostic PRM (not used by the method)")
    g.add_argument("--verifier_model", default="Qwen/Qwen2.5-Math-PRM-7B")
    g.add_argument("--prm_aggregation", default="min", choices=["min", "mean", "last"])
    g.add_argument("--no_prm", action="store_true",
                   help="Skip loading the PRM entirely")

    g = p.add_argument_group("Dataset")
    g.add_argument("--dataset", required=True, choices=DATASETS)
    g.add_argument("--split", default=None)
    g.add_argument("--max_samples", type=int, default=None)
    g.add_argument("--custom_path", default=None)

    g = p.add_argument_group("Breadth-depth refinement")
    g.add_argument("--cimr_n_rollouts", type=int, default=8, help="Breadth N")
    g.add_argument("--cimr_depth", type=int, default=4, help="Depth D")
    g.add_argument("--no_critique", action="store_true",
                   help="Ablation: skip self-critique")
    g.add_argument("--no_correction", action="store_true",
                   help="Ablation: skip self-correction")
    g.add_argument("--cimr_reasoning_max_new_tokens", type=int, default=2048)
    g.add_argument("--cimr_reasoning_temperature", type=float, default=0.7,
                   help="Rollout sampling temperature (depth 1)")
    g.add_argument("--cimr_reasoning_top_p", type=float, default=0.95)
    g.add_argument("--cimr_critique_max_tokens", type=int, default=384)
    g.add_argument("--cimr_critique_max_trace_chars", type=int, default=4000)
    g.add_argument("--cimr_correction_max_tokens", type=int, default=1024)
    g.add_argument("--cimr_correction_max_trace_chars", type=int, default=4000)
    g.add_argument("--cimr_correction_max_critique_chars", type=int, default=1500)
    g.add_argument("--cimr_continuation_max_prior_chars", type=int, default=4000)

    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", default="./results_paired")
    g.add_argument("--log_every", type=int, default=10,
                   help="Write the summary and log accuracy every K questions")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy summary
# ─────────────────────────────────────────────────────────────────────────────

def summarize(question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Accuracy per selection rule over questions with a correctness label."""
    summary: Dict[str, Any] = {}
    for rule in SELECTION_RULES:
        key = f"cimr_{rule}"
        flags = [
            qr[key]["is_correct"] for qr in question_results
            if key in qr and qr[key].get("is_correct") is not None
        ]
        if flags:
            n_correct = sum(int(bool(f)) for f in flags)
            summary[key] = dict(
                accuracy=100.0 * n_correct / len(flags),
                n_correct=n_correct,
                n_total=len(flags),
            )
    return summary


def _write_outputs(summary_path: str, results_path: str,
                   question_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Atomically write the summary JSON and the per-question JSONL."""
    summary = summarize(question_results)
    tmp = summary_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, summary_path)

    tmp = results_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for qr in question_results:
            f.write(json.dumps(qr, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, results_path)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tag_suffix = ""
    if args.no_critique:
        tag_suffix += "_no_critique"
    if args.no_correction:
        tag_suffix += "_no_correction"
    tag = f"{args.dataset}_N{args.cimr_n_rollouts}_d{args.cimr_depth}{tag_suffix}"

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"CUDA_VISIBLE_DEVICES={args.gpus}")

    from src.cimr_inference import CIMRConfig, CIMRInferencer
    from src.cimr_step import CIMRStepConfig
    from src.dataset_loader import load_dataset_split
    from src.inference_methods import _build_evaluator
    from src.model_wrapper import HFModelWrapper
    from src.verifier import build_verifier

    logger.info(f"Loading dataset: {args.dataset}")
    examples = load_dataset_split(
        data_name=args.dataset, split=args.split,
        max_samples=args.max_samples, custom_path=args.custom_path,
        cache_dir=args.hf_cache_dir,
    )
    data_name = examples[0]["_data_name"] if examples else args.dataset

    logger.info(f"Loading generator: {args.model}")
    model = HFModelWrapper(
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        device=args.device, dtype=args.dtype,
        load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.hf_cache_dir,
    )

    verifier = None
    if not args.no_prm:
        logger.info(f"Loading diagnostic PRM: {args.verifier_model}")
        verifier = build_verifier(
            verifier_type="prm", data_name=data_name,
            prm_model=args.verifier_model,
            prm_aggregation=args.prm_aggregation,
            device=args.device, cache_dir=args.hf_cache_dir,
            batch_size=args.batch_size,
        )

    ev = _build_evaluator(data_name)

    N, D = int(args.cimr_n_rollouts), int(args.cimr_depth)
    logger.info(
        f"Refinement  N={N}  D={D}  no_critique={args.no_critique}  "
        f"no_correction={args.no_correction}  prm={'off' if verifier is None else 'on'}"
    )
    step_cfg = CIMRStepConfig(
        reasoning_max_new_tokens=args.cimr_reasoning_max_new_tokens,
        reasoning_temperature=args.cimr_reasoning_temperature,
        reasoning_top_p=args.cimr_reasoning_top_p,
        reasoning_do_sample=True,
        critique_max_tokens=args.cimr_critique_max_tokens,
        critique_max_trace_chars=args.cimr_critique_max_trace_chars,
        correction_max_tokens=args.cimr_correction_max_tokens,
        correction_max_trace_chars=args.cimr_correction_max_trace_chars,
        correction_max_critique_chars=args.cimr_correction_max_critique_chars,
        continuation_max_prior_chars=args.cimr_continuation_max_prior_chars,
        no_critique=args.no_critique,
        no_correction=args.no_correction,
    )
    inferencer = CIMRInferencer(
        model=model, verifier=verifier, ev=ev, data_name=data_name,
        config=CIMRConfig(n_rollouts=N, depth=D, step_config=step_cfg,
                          tag_suffix=tag_suffix),
        output_dir=args.output_dir, dataset=args.dataset,
    )

    summary_path = os.path.join(args.output_dir, f"summary_paired_{tag}.json")
    results_path = os.path.join(args.output_dir, f"results_paired_{tag}.jsonl")

    question_results: List[Dict[str, Any]] = []
    n_resumed = n_fresh = 0
    for i, ex in enumerate(examples):
        qid = str(ex.get("id", i))
        if (inferencer.store.get(qid) or {}).get("completed", False):
            n_resumed += 1
        else:
            n_fresh += 1

        qrec = inferencer.run_question(
            question=ex["question"], gold_answer=ex.get("answer"), qid=qid,
        )
        agg = qrec.get("aggregation") or {}
        question_results.append({
            "__id__": qid,
            "cimr_majority": dict(
                method="cimr_majority", selection_rule="majority",
                predicted_answer=agg.get("majority_pred", ""),
                is_correct=agg.get("majority_is_correct"),
                rollout_idx=agg.get("majority_rollout_idx", -1),
                vote_count=agg.get("majority_vote_count", 0),
                n_rollouts=agg.get("n_rollouts", N), depth=D,
            ),
            "cimr_max_prm": dict(
                method="cimr_max_prm", selection_rule="max_prm",
                predicted_answer=agg.get("max_prm_pred", ""),
                is_correct=agg.get("max_prm_is_correct"),
                rollout_idx=agg.get("max_prm_rollout_idx", -1),
                prm_score=agg.get("max_prm_score", 0.0),
                n_rollouts=agg.get("n_rollouts", N), depth=D,
            ),
        })

        if (i + 1) % max(args.log_every, 1) == 0 or (i + 1) == len(examples):
            try:
                summary = _write_outputs(summary_path, results_path, question_results)
            except Exception as e:
                logger.warning(f"Failed to write incremental outputs: {e}")
                summary = summarize(question_results)
            logger.info(
                f"  q={i + 1}/{len(examples)}  fresh={n_fresh}  resumed={n_resumed}  "
                f"id={qid}  maj={agg.get('majority_pred', '')!r}"
                f"({agg.get('majority_is_correct')})"
            )
            for m, v in sorted(summary.items()):
                logger.info(
                    f"    {m:<16s} acc={v['accuracy']:6.2f}%  "
                    f"({v['n_correct']}/{v['n_total']})"
                )

    summary = _write_outputs(summary_path, results_path, question_results)

    logger.info("=" * 72)
    logger.info(
        f" SUMMARY  {args.dataset}  N={N}  D={D}  ablation={tag_suffix or 'none'}"
    )
    for rule in SELECTION_RULES:
        s = summary.get(f"cimr_{rule}")
        if s:
            logger.info(
                f"  {rule:<10s} acc={s['accuracy']:6.2f}%  ({s['n_correct']}/{s['n_total']})"
            )
    logger.info(f"  store:   {inferencer.store_path}")
    logger.info(f"  summary: {summary_path}")
    logger.info(f"  results: {results_path}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
