# Refining Over Resampling: Test-Time Self-Correction for LLM Reasoning

Code for a **training-free, verifier-free breadth–depth test-time refinement**
framework for mathematical reasoning. For each problem, one base model

1. samples **N** independent reasoning rollouts (*breadth*), which preserves
   diverse solution directions;
2. refines every rollout through **D** rounds of reasoning continuation →
   self-critique → self-correction (*depth*), which repairs per-trace errors;
3. extracts one answer per refined rollout and returns their **plurality vote**.

No reward model, learned stopping rule or extra training is involved. The
same model plays all three roles through role-specific prompts.

> In the code the method is called **CIMR** (`src/cimr_*.py`, `--cimr_*`
> flags, `cimr_full_results_*.json`, `cimr_majority`). It is the same method
> the paper calls breadth–depth refinement or "Ours".

---

## Method and where it lives in the code

| Paper (Algorithm 1) | Code |
|---|---|
| Generator π<sup>G</sup>, critic π<sup>C</sup>, corrector π<sup>R</sup> role prompts | `src/cimr_prompts.py` |
| Initial rollout r<sub>i</sub><sup>(0)</sup> ~ π<sup>G</sup>(· \| x) at temperature τ | `run_step_batch`, depth 1, in `src/cimr_step.py` |
| Reasoning continuation g<sub>i</sub><sup>(d)</sup> ~ π<sup>G</sup>(· \| x, r<sub>i</sub><sup>(d−1)</sup>) | `run_step_batch`, stage 1 |
| Self-critique c<sub>i</sub><sup>(d)</sup> ~ π<sup>C</sup>(· \| x, g<sub>i</sub><sup>(d)</sup>) | `run_step_batch`, stage 2 |
| Self-correction r<sub>i</sub><sup>(d)</sup> ~ π<sup>R</sup>(· \| x, g<sub>i</sub><sup>(d)</sup>, c<sub>i</sub><sup>(d)</sup>) | `run_step_batch`, stage 3 |
| N rollouts × D depths, batched across rollouts | `run_all_rollouts_batched` in `src/cimr_rollout.py` |
| â<sub>i</sub> = Extract(r<sub>i</sub><sup>(D)</sup>), â = Vote(â<sub>1..N</sub>) | `extract_answer` (`cimr_step.py`), `aggregate_rollouts` (`src/cimr_aggregator.py`) |
| Per-problem orchestration and resumable storage | `CIMRInferencer` in `src/cimr_inference.py` |

**Depth indexing.** At depth 1 there is no previous corrected trace, so the
generator samples from the problem alone. That sample is the initial rollout
r<sup>(0)</sup>, and it is critiqued and corrected directly (it also serves as
g<sup>(1)</sup>). From depth 2 onwards the generator continues from
r<sup>(d−1)</sup>. A run with depth D makes **3·N·D** generation calls in
**3·D** batched rounds per problem.

---

## Repository layout

```text
.
├── run_paired_ablation.py    # breadth–depth refinement on one benchmark
├── run_paired_ablation.sh    # one model over several benchmarks (paper defaults)
├── run_nd_sweep.sh           # sweep over N and D from a config file
├── batch_nd_sweep.sh         # Slurm wrapper for run_nd_sweep.sh
├── run_eval.py               # baselines: Greedy, Maj@N, Best-of-N, log-prob selection
├── configs/                  # sweep configs, one per model (+ w/o-critic variants)
├── src/
│   ├── cimr_prompts.py       # role prompts (reproduced verbatim in the paper)
│   ├── cimr_step.py          # one refinement depth
│   ├── cimr_rollout.py       # N rollouts × D depths
│   ├── cimr_aggregator.py    # answer normalisation and plurality vote
│   ├── cimr_inference.py     # orchestration and the results store
│   ├── model_wrapper.py      # Hugging Face generation wrapper
│   ├── dataset_loader.py     # benchmarks and gold answers
│   ├── evaluator_classes.py  # answer extraction / equivalence per benchmark
│   ├── math_toolkit/         # symbolic answer checking (vendored, MIT)
│   ├── inference_methods.py  # baseline methods
│   ├── evaluator.py          # resumable evaluation loop for baselines
│   ├── verifier.py           # PRM scorer (Best-of-N baseline, diagnostics)
│   └── clustering.py         # sampling-redundancy clustering (motivation analysis)
├── extract_cimr_metrics.py   # refinement-dynamics metrics (appendix definitions)
├── export_paper_csvs.py      # metrics → CSVs for plotting
├── plot_cimr_metrics.m       # MATLAB figures from those CSVs
├── extract_compute_flop.py   # TFLOP accounting for refinement runs
├── extract_baseline_flop.py  # TFLOP accounting for baseline runs
└── print_paired_accuracies.py
```

---

## Installation

Python 3.12. The pinned environment is in `pyproject.toml` / `uv.lock`:

```bash
uv sync
```

Or install the minimal requirements with pip:

```bash
pip install -r requirements.txt
pip install bitsandbytes>=0.43.0   # only for --load_in_4bit / --load_in_8bit
```

Models and datasets are downloaded from the Hugging Face Hub. Set `HF_HOME` to
choose the cache location.

---

## Running breadth–depth refinement

### One model, several benchmarks

```bash
bash run_paired_ablation.sh \
  --model_name Qwen/Qwen2.5-Math-7B-Instruct \
  --cimr_n_rollouts 8 --cimr_depth 4 \
  aime24 aime25 amc23 math500 olympiadbench
```

Benchmarks default to `aime24 aime25 amc23 math500 olympiadbench`; `gsm8k` is
also accepted. Set `PYTHON="uv run python"` or `GPUS=0` in the environment if
needed.

### Ablations

| Flag | Effect | Paper |
|---|---|---|
| `--no_critique` | the critic is skipped; the corrector gets an empty critique (its CRITIQUE slot shows a fixed "no specific errors identified" line) | w/o row of the self-critic ablation |
| `--no_correction` | the corrector is skipped; r<sup>(d)</sup> = g<sup>(d)</sup> | — |
| `--no_prm` | the diagnostic PRM is not loaded (saves a 7B model's memory and compute) | — |

### Sweeping N and D

Each file in `configs/` sets `MODEL_NAME`, `N_VALUES`, `D_VALUES`, `DATASETS`,
`NO_CRITIQUE` and `NO_CORRECTION`. `configs/<model>.conf` runs the full
pipeline and `configs/<model>_no_critique.conf` the w/o-critic ablation.

```bash
bash run_nd_sweep.sh configs/qwen_7b.conf            # runs through `uv run` by default
RUNNER="" bash run_nd_sweep.sh configs/qwen_7b.conf  # use the active environment
sbatch batch_nd_sweep.sh configs/qwen_7b.conf        # Slurm; add your partition directives
```

| Config | Model |
|---|---|
| `qwen_7b` | Qwen/Qwen2.5-Math-7B-Instruct |
| `qwen_1b` | Qwen/Qwen2.5-1.5B-Instruct |
| `ministral_8b` | mistralai/Ministral-8B-Instruct-2410 |
| `llama3_8b` | meta-llama/Llama-3.1-8B-Instruct |

### Python entry point

```bash
python run_paired_ablation.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset amc23 \
  --cimr_n_rollouts 8 --cimr_depth 4 --output_dir results/ [--no_prm]
```

Run `python run_paired_ablation.py -h` for all options (token budgets,
truncation limits, quantisation, device placement).

---

## Baselines

```bash
python run_eval.py --model Qwen/Qwen2.5-Math-7B-Instruct --dataset amc23 \
  --methods greedy bon_majority bon greedy_pass_n --N 8 --batch_size 64
```

| `--methods` | Baseline |
|---|---|
| `greedy` | Greedy decoding (τ = 0) |
| `bon_majority` | Majority vote, Maj@N (self-consistency) |
| `bon` | Best-of-N with a verifier: the highest-scoring sample under `--prm_model` (default Qwen/Qwen2.5-Math-PRM-7B) |
| `greedy_pass_n` | N samples; keep the one with the highest mean token log-probability |

All baselines share one prompt (`MATH_SYSTEM_PROMPT` in
`src/model_wrapper.py`) and the same answer extraction and correctness checks
as the refinement pipeline. Beam search and Lookahead are not implemented in
this repository.

---

## Outputs

`run_paired_ablation.sh` writes to
`experiments_results/results_paired/<model>_mnt2048_t0.7_p0.95/`. With
`<tag> = <dataset>_N<N>_d<D>[_no_critique][_no_correction]`:

| File | Content |
|---|---|
| `cimr_full_results_<tag>.json` | Full store: per question, per rollout, per depth: g<sup>(d)</sup>, its answer, c<sup>(d)</sup>, r<sup>(d)</sup>, its answer, diagnostic PRM scores, plus the aggregation. Rewritten after every (rollout, depth) step; re-running the same command resumes from it. |
| `summary_paired_<tag>.json` | Accuracy of `cimr_majority` (the method) and `cimr_max_prm` (diagnostic) |
| `results_paired_<tag>.jsonl` | One prediction record per question |
| `logs/paired_<tag>.log` | Console log |

`--no_prm` does not change the tag, so a `--no_prm` run into the same folder
resumes from an existing store.

`run_eval.py` writes `<method>_<dataset>_N<N>.jsonl` (resumable),
`<method>_<dataset>_N<N>_summary.json` and `comparison_<dataset>_N<N>.txt`.

---

## Analysis

```bash
# Accuracy table across runs
python print_paired_accuracies.py --results_dir experiments_results/results_paired

# Refinement dynamics: depth-wise accuracy A(d), recovery ρ_rec(d), regression ρ_reg(d),
# answer-change rate μ_Δ(d), flip rates μ_W→C(d) / μ_C→W(d), net benefit B(d),
# terminal diversity and answer agreement U(d)
python extract_cimr_metrics.py --results_dir experiments_results/results_paired \
                               --output_dir  experiments_results/cimr_metrics
python export_paper_csvs.py    # N=8, D=4 CSVs → experiments_results/paper_csvs/
matlab -batch plot_cimr_metrics

# Compute: 2·P·(T_in + T_out) per generation call, 2·P_rm·T_in per verifier pass
python extract_compute_flop.py  --results_dir experiments_results/results_paired
python extract_baseline_flop.py --results_dir experiments_results/results_baselines
```

Token counts use tiktoken's `cl100k_base` as a proxy for each model's
tokenizer. For the refinement method, the verifier-free compute is
`generator_tflop_per_question`; `total_tflop_per_question` also includes the
optional diagnostic PRM passes.

The sampling-redundancy statistics of the motivation analysis (realised
clusters R(N), dominant-cluster mass f<sub>max</sub>(N), entropy H(N)) are
implemented by `ClusterManager` and `redundancy_metrics` in `src/clustering.py`
(all-MiniLM-L6-v2 embeddings, average-linkage cosine clustering,
τ<sub>sim</sub> = 0.85).

---

## Implementation details

**Decoding.** Depth-1 rollouts are sampled with temperature 0.7, top-p 0.95
and top-k 50. Reasoning continuation (d ≥ 2), critique and correction use
greedy decoding.

**Token budgets** (`run_paired_ablation.sh` defaults). Generator: 2048 new
tokens. Critic: 384. Corrector: 1024. Inside prompts, traces longer than
4000 characters keep their head and tail, and critiques are cut to their first
1500 characters. The model wrapper truncates prompts to 2048 tokens.

**Answer extraction.** Traces are stored and passed on in full. `Extract(·)`
returns the content of the last `\boxed{}` in a trace, or the last number if
no box is present. If the corrector returns nothing, g<sup>(d)</sup> is carried
forward as r<sup>(d)</sup>.

**Voting.** Answers are grouped after light normalisation (whitespace and
LaTeX spacing removed, one brace layer unwrapped, trailing decimal zeros
stripped), and empty answers do not vote. Ties on vote count are broken by the
larger summed diagnostic PRM score of the tied group, then by rollout order;
with `--no_prm`, by rollout order only. Correctness against the gold answer
uses the benchmark evaluator (string, LaTeX-normalised, symbolic and numeric
equivalence).

**Diagnostic PRM.** Unless `--no_prm` is set, Qwen2.5-Math-PRM-7B scores
g<sup>(d)</sup> and r<sup>(d)</sup> at every depth. The scores are stored for
analysis and reported as `cimr_max_prm`; apart from the tie-break above they
do not influence the method.

**Benchmarks.**

| `--dataset` | Hugging Face source | Problems used |
|---|---|---|
| `aime24` | Maxwell-Jia/AIME_2024 | 30 |
| `aime25` | math-ai/aime25 | 30 |
| `amc23` | math-ai/amc23 | 40 |
| `math500` | HuggingFaceH4/MATH-500 | 500 |
| `olympiadbench` | math-ai/olympiadbench | full test split |
| `gsm8k` | openai/gsm8k | full test split |
| `custom` | local JSON/JSONL via `--custom_path` | — |

`src/math_toolkit/` is adapted from Xwin-Math (MIT License), which builds on
ToRA's grader.
