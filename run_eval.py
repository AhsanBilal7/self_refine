"""Evaluate the sampling baselines on one benchmark.

Methods (``--methods``)
-----------------------
  greedy         Greedy decoding (tau = 0)
  bon_majority   Maj@N: N samples, plurality vote
  bon            Best-of-N: N samples, highest verifier score (``--prm_model``)
  greedy_pass_n  N samples, highest mean token log-probability

All methods share the baseline prompt (``src.model_wrapper.MATH_SYSTEM_PROMPT``)
and the answer extraction / correctness checks of the refinement pipeline.

Outputs (in --output_dir), one pair per method
----------------------------------------------
  <method>_<dataset>_N<N>.jsonl          one row per question (resumable)
  <method>_<dataset>_N<N>_summary.json   running accuracy summary
  comparison_<dataset>_N<N>.txt          accuracy table across methods

Example
-------
  python run_eval.py --model Qwen/Qwen2.5-Math-7B-Instruct --dataset amc23 \\
      --methods greedy bon_majority bon --N 8 --batch_size 64
"""

import argparse
import logging
import math
import os

from src.dataset_loader import DATASETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_eval")

METHODS = ["greedy", "bon_majority", "bon", "greedy_pass_n"]
SAMPLING_METHODS = {"bon_majority", "bon", "greedy_pass_n"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Model")
    g.add_argument("--model", required=True, help="Hugging Face model id or local path")
    g.add_argument("--batch_size", type=int, default=8,
                   help="Sequences per generate() call; use a multiple of --N")
    g.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    g.add_argument("--device", default="auto",
                   help="'auto' (all visible GPUs) | 'cuda:N' | 'cpu'")
    g.add_argument("--gpus", default=None, help="Sets CUDA_VISIBLE_DEVICES, e.g. '0,1'")
    g.add_argument("--load_in_4bit", action="store_true")
    g.add_argument("--load_in_8bit", action="store_true")
    g.add_argument("--max_new_tokens", type=int, default=1024)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--top_p", type=float, default=0.95)
    g.add_argument("--trust_remote_code", action="store_true")
    g.add_argument("--hf_cache_dir", default=None)

    g = p.add_argument_group("Dataset")
    g.add_argument("--dataset", required=True, choices=DATASETS)
    g.add_argument("--split", default=None)
    g.add_argument("--max_samples", type=int, default=None)
    g.add_argument("--custom_path", default=None)

    g = p.add_argument_group("Methods")
    g.add_argument("--methods", nargs="+", default=["greedy", "bon_majority"],
                   choices=METHODS)
    g.add_argument("--N", type=int, default=8, help="Samples per question")

    g = p.add_argument_group("Verifier (bon only)")
    g.add_argument("--prm_model", default="Qwen/Qwen2.5-Math-PRM-7B")
    g.add_argument("--prm_aggregation", default="min", choices=["min", "mean", "last"])

    g = p.add_argument_group("Output")
    g.add_argument("--output_dir", default="./results_baselines")
    g.add_argument("--no_resume", action="store_true")
    g.add_argument("--log_every", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Must be set before torch initialises CUDA.
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"CUDA_VISIBLE_DEVICES={args.gpus}")

    from src.dataset_loader import load_dataset_split
    from src.evaluator import compare_methods, evaluate_batch, redundancy_stats, save_summary
    from src.inference_methods import build_inference_method
    from src.model_wrapper import HFModelWrapper
    from src.verifier import build_verifier

    if set(args.methods) & SAMPLING_METHODS and args.batch_size < args.N:
        logger.warning(
            f"batch_size={args.batch_size} < N={args.N}: each question's samples "
            f"span {math.ceil(args.N / args.batch_size)} generate() calls."
        )

    examples = load_dataset_split(
        data_name=args.dataset, split=args.split,
        max_samples=args.max_samples, custom_path=args.custom_path,
        cache_dir=args.hf_cache_dir,
    )
    data_name = examples[0]["_data_name"] if examples else args.dataset

    model = HFModelWrapper(
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.hf_cache_dir,
    )

    verifier = None
    if "bon" in args.methods:
        verifier = build_verifier(
            verifier_type="prm", data_name=data_name,
            prm_model=args.prm_model, prm_aggregation=args.prm_aggregation,
            device=args.device, cache_dir=args.hf_cache_dir,
            batch_size=args.batch_size,
        )

    tag = f"{args.dataset}_N{args.N}"
    all_summaries = {}
    for method_name in args.methods:
        method = build_inference_method(
            method_name=method_name, model=model, verifier=verifier,
            N=args.N, data_name=data_name,
            temperature=args.temperature, top_p=args.top_p,
        )
        logger.info(f"Running {method_name} on {args.dataset}  N={args.N}")

        summary_path = os.path.join(args.output_dir, f"{method_name}_{tag}_summary.json")
        summary = evaluate_batch(
            method=method,
            examples=examples,
            max_new_tokens=args.max_new_tokens,
            save_path=os.path.join(args.output_dir, f"{method_name}_{tag}.jsonl"),
            summary_path=summary_path,
            resume=not args.no_resume,
            log_every=args.log_every,
        )
        if method_name == "bon_majority":
            summary["redundancy_stats"] = redundancy_stats(summary["results"])
            logger.info(f"Vote-group stats: {summary['redundancy_stats']}")
            save_summary(summary, summary_path)
        all_summaries[method_name] = summary

    comparison = compare_methods(all_summaries)
    logger.info(comparison)
    cmp_path = os.path.join(args.output_dir, f"comparison_{tag}.txt")
    with open(cmp_path, "w") as f:
        f.write(comparison)
    logger.info(f"Comparison table saved to {cmp_path}")


if __name__ == "__main__":
    main()
