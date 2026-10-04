"""Baseline test-time inference methods (``run_eval.py``).

All methods use the same base model, the same baseline prompt
(``model_wrapper.MATH_SYSTEM_PROMPT``) and the same answer extraction and
correctness checking as the breadth-depth refinement pipeline.

====================  =================  ======================================
class                 ``--methods`` key  selection
====================  =================  ======================================
``GreedyInference``   ``greedy``         one greedy decode (tau = 0)
``BestOfNMajority``   ``bon_majority``   N samples, plurality vote (Maj@N)
``BestOfN``           ``bon``            N samples, highest verifier score
                                         (Best-of-N with a reward model)
``GreedyPassN``       ``greedy_pass_n``  N samples, highest mean token
                                         log-probability under the model
====================  =================  ======================================

Each method builds its dataset ``Evaluator`` once in ``__init__`` and reuses it
for every question. ``_extract_pred`` and ``_check_correct`` are also used by
the refinement pipeline to score its predictions.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Outcome of one method on one question (one JSONL row via ``to_dict``)."""

    method:           str
    question:         str
    gold_answer:      Optional[str]
    selected_trace:   str
    predicted_answer: str
    is_correct:       Optional[bool]
    N:                int
    elapsed_sec:      float

    all_traces:        List[str]   = field(default_factory=list)
    raw_scores:        List[float] = field(default_factory=list)
    normalized_scores: List[float] = field(default_factory=list)
    cluster_labels:    List[int]   = field(default_factory=list)
    cluster_sizes:     List[int]   = field(default_factory=list)
    selected_index:    int         = -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method":             self.method,
            "gold_answer":        self.gold_answer,
            "predicted_answer":   self.predicted_answer,
            "raw_llm_output":     self.selected_trace,
            "is_correct":         self.is_correct,
            "N":                  self.N,
            "elapsed_sec":        round(self.elapsed_sec, 3),
            "selected_index":     self.selected_index,
            # Vote groups (bon_majority): group id and group size per trace.
            "cluster_labels":     self.cluster_labels,
            "cluster_sizes":      self.cluster_sizes,
            # Selection score per trace: verifier score (bon), log-prob
            # (greedy_pass_n) or vote count (bon_majority).
            "raw_scores":         [round(s, 4) for s in self.raw_scores],
            "normalized_scores":  [round(s, 4) for s in self.normalized_scores],
            # Character length of every candidate; extract_baseline_flop.py
            # uses these to estimate generated tokens for unselected traces.
            "trace_lengths":      [len(t) for t in self.all_traces] if self.all_traces else [],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator factory
# ──────────────────────────────────────────────────────────────────────────────

# dataset name -> task key understood by evaluator_classes.get_evaluator()
_EVALUATOR_MAP: Dict[str, str] = {
    "math500":       "MATH500",
    "math":          "MATH500",
    "aime24":        "AIME2024",
    "aime25":        "AIME",
    "amc23":         "AMC2023",
    "gsm8k":         "GSM8K",
    "olympiadbench": "OlympiadBench",
    "custom":        "MATH500",
}


def _build_evaluator(data_name: str):
    """Return the ``Evaluator`` for ``data_name`` (``None`` if it cannot be built)."""
    try:
        from .evaluator_classes import get_evaluator  # type: ignore
        task = _EVALUATOR_MAP.get(data_name, "MATH500")
        ev = get_evaluator(task)
        logger.info(f"Evaluator: {ev.__class__.__name__} for dataset='{data_name}'")
        return ev
    except Exception as e:
        logger.warning(
            f"Could not build Evaluator for '{data_name}': {e}. "
            f"Falling back to string comparison."
        )
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Extraction and correctness helpers  (use the pre-built evaluator)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_pred(trace: str, ev, data_name: str) -> str:
    """Extract the final answer from a model trace.

    Uses the evaluator's ``\\boxed{}`` extraction (without requiring a box to
    be present) and falls back to the last number in the trace.
    """
    if ev is not None:
        try:
            ans = ev._extract_answer_from_model_completion(trace)
            if ans is not None and ans != "":
                return ans
        except Exception:
            pass

    nums = re.findall(r"-?\d+(?:\.\d+)?", trace.replace(",", ""))
    return nums[-1] if nums else ""


def _check_correct(pred: str, gold: Optional[str], ev, data_name: str) -> Optional[bool]:
    """Compare a prediction with the gold answer.

    ``None`` without a gold answer and ``False`` for an empty prediction;
    otherwise the evaluator's ``check_answers_equiv``, falling back to
    stripped string equality if the evaluator is unavailable or undecided.
    """
    if gold is None:
        return None
    if not pred:
        return False

    if ev is not None:
        try:
            result = ev.check_answers_equiv(pred, gold)
            if result is not None:
                return bool(result)
        except Exception:
            pass

    return pred.strip() == gold.strip()


def _make_error_result(method_name: str, question: str,
                        gold: Optional[str], N: int) -> InferenceResult:
    return InferenceResult(
        method=method_name, question=question, gold_answer=gold,
        selected_trace="", predicted_answer="", is_correct=False,
        N=N, elapsed_sec=0.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Greedy Inference
# ──────────────────────────────────────────────────────────────────────────────

class GreedyInference:
    """Greedy baseline: one deterministic decode per question (tau = 0)."""

    def __init__(self, model, data_name: str = "math"):
        self.model     = model
        self.data_name = data_name
        self.N         = 1
        self.ev        = _build_evaluator(data_name)

    def run(
        self,
        question: str,
        gold_answer: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> InferenceResult:
        t0      = time.time()
        trace   = self.model.generate_greedy(question, max_new_tokens=max_new_tokens)
        elapsed = time.time() - t0

        pred    = _extract_pred(trace, self.ev, self.data_name)
        correct = _check_correct(pred, gold_answer, self.ev, self.data_name)

        return InferenceResult(
            method="greedy", question=question,
            gold_answer=gold_answer, selected_trace=trace,
            predicted_answer=pred, is_correct=correct,
            N=1, elapsed_sec=elapsed,
            all_traces=[trace], selected_index=0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Best-of-N with a verifier
# ──────────────────────────────────────────────────────────────────────────────

class BestOfN:
    """Best-of-N: sample N traces and keep ``argmax_i V(x_i)``.

    ``V`` is the verifier built by ``verifier.build_verifier``.
    """

    def __init__(
        self,
        model,
        verifier,
        N: int = 8,
        data_name: str = "math",
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.model       = model
        self.verifier    = verifier
        self.N           = N
        self.data_name   = data_name
        self.temperature = temperature
        self.top_p       = top_p
        self.ev          = _build_evaluator(data_name)

    def _select(
        self,
        question: str,
        gold: Optional[str],
        traces: List[str],
        elapsed: float,
    ) -> InferenceResult:
        from .verifier import score_candidates

        scores   = score_candidates(traces, self.verifier, gold, question=question)
        best_idx = int(np.argmax(scores))
        selected = traces[best_idx]

        pred    = _extract_pred(selected, self.ev, self.data_name)
        correct = _check_correct(pred, gold, self.ev, self.data_name)

        return InferenceResult(
            method="bon", question=question,
            gold_answer=gold, selected_trace=selected,
            predicted_answer=pred, is_correct=correct,
            N=self.N, elapsed_sec=elapsed,
            all_traces=traces,
            raw_scores=scores.tolist(),
            normalized_scores=scores.tolist(),
            selected_index=best_idx,
        )

    def run(
        self,
        question: str,
        gold_answer: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> InferenceResult:
        t0      = time.time()
        traces  = self.model.generate_samples(
            question, N=self.N,
            temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0
        return self._select(question, gold_answer, traces, elapsed)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Log-probability selection
# ──────────────────────────────────────────────────────────────────────────────

class GreedyPassN:
    """Sample N traces and keep the one with the highest mean token log-prob.

    Needs no reward model; selection costs one forward pass per candidate.
    """

    def __init__(
        self,
        model,
        N: int = 8,
        data_name: str = "math",
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.model       = model
        self.N           = N
        self.data_name   = data_name
        self.temperature = temperature
        self.top_p       = top_p
        self.ev          = _build_evaluator(data_name)

    def _select(
        self,
        question: str,
        gold: Optional[str],
        traces: List[str],
        elapsed: float,
    ) -> InferenceResult:
        prompt   = self.model._build_prompt(question)
        logprobs = self.model.score_sequences_logprob(prompt, traces)
        best_idx = int(np.argmax(logprobs))
        selected = traces[best_idx]

        pred    = _extract_pred(selected, self.ev, self.data_name)
        correct = _check_correct(pred, gold, self.ev, self.data_name)

        return InferenceResult(
            method="greedy_pass_n", question=question,
            gold_answer=gold, selected_trace=selected,
            predicted_answer=pred, is_correct=correct,
            N=self.N, elapsed_sec=elapsed,
            all_traces=traces,
            raw_scores=logprobs,
            normalized_scores=logprobs,
            selected_index=best_idx,
        )

    def run(
        self,
        question: str,
        gold_answer: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> InferenceResult:
        t0      = time.time()
        traces  = self.model.generate_samples(
            question, N=self.N,
            temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0
        return self._select(question, gold_answer, traces, elapsed)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Majority vote (self-consistency, Maj@N)
# ──────────────────────────────────────────────────────────────────────────────

class BestOfNMajority:
    """Maj@N: sample N traces and return the plurality answer.

    Answers are grouped with ``ev.check_answers_equiv`` so LaTeX variants of
    the same value share a vote; the first trace of the largest group is
    returned. Ties go to the group seen first.
    """

    def __init__(
        self,
        model,
        N: int = 8,
        data_name: str = "math",
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.model       = model
        self.N           = N
        self.data_name   = data_name
        self.temperature = temperature
        self.top_p       = top_p
        self.ev          = _build_evaluator(data_name)

    def _majority_vote(self, traces: List[str]):
        """
        Group traces by equivalent final answer.

        Returns
        -------
        best_idx      : index of selected trace (first in majority group)
        majority_ans  : the majority answer string
        vote_counts   : vote_counts[i] = size of trace i's group
        group_labels  : group id for each trace
        """
        answers = [_extract_pred(t, self.ev, self.data_name) for t in traces]

        groups: List[List[int]] = []
        group_rep: List[str]    = []

        for idx, ans in enumerate(answers):
            placed = False
            for g, rep in enumerate(group_rep):
                try:
                    equiv = (
                        self.ev.check_answers_equiv(ans, rep)
                        if self.ev is not None else None
                    )
                except Exception:
                    equiv = None
                if equiv is None:
                    equiv = (ans.strip() == rep.strip())
                if equiv:
                    groups[g].append(idx)
                    placed = True
                    break
            if not placed:
                groups.append([idx])
                group_rep.append(ans)

        majority_g   = max(range(len(groups)), key=lambda g: len(groups[g]))
        best_idx     = groups[majority_g][0]
        majority_ans = group_rep[majority_g]

        vote_counts  = [0] * len(traces)
        group_labels = [0] * len(traces)
        for g, members in enumerate(groups):
            for idx in members:
                vote_counts[idx]  = len(members)
                group_labels[idx] = g

        return best_idx, majority_ans, vote_counts, group_labels

    def _select(
        self,
        question: str,
        gold: Optional[str],
        traces: List[str],
        elapsed: float,
    ) -> InferenceResult:
        best_idx, majority_ans, vote_counts, group_labels = \
            self._majority_vote(traces)

        selected = traces[best_idx]
        pred     = majority_ans or _extract_pred(selected, self.ev, self.data_name)
        correct  = _check_correct(pred, gold, self.ev, self.data_name)

        max_v      = max(vote_counts) if vote_counts else 1
        norm_votes = [v / max_v for v in vote_counts]

        return InferenceResult(
            method="bon_majority", question=question,
            gold_answer=gold, selected_trace=selected,
            predicted_answer=pred, is_correct=correct,
            N=self.N, elapsed_sec=elapsed,
            all_traces=traces,
            raw_scores=[float(v) for v in vote_counts],
            normalized_scores=norm_votes,
            cluster_labels=group_labels,
            cluster_sizes=vote_counts,
            selected_index=best_idx,
        )

    def run(
        self,
        question: str,
        gold_answer: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> InferenceResult:
        t0      = time.time()
        traces  = self.model.generate_samples(
            question, N=self.N,
            temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0
        return self._select(question, gold_answer, traces, elapsed)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_inference_method(
    method_name: str,
    model,
    verifier=None,
    N: int = 8,
    data_name: str = "math",
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    """Instantiate a baseline by its ``--methods`` key (see module docstring)."""
    if method_name == "greedy":
        return GreedyInference(model=model, data_name=data_name)
    if method_name == "bon":
        if verifier is None:
            raise ValueError("Method 'bon' requires a verifier.")
        return BestOfN(model=model, verifier=verifier, N=N,
                       data_name=data_name, temperature=temperature, top_p=top_p)
    if method_name == "greedy_pass_n":
        return GreedyPassN(model=model, N=N, data_name=data_name,
                           temperature=temperature, top_p=top_p)
    if method_name == "bon_majority":
        return BestOfNMajority(model=model, N=N, data_name=data_name,
                               temperature=temperature, top_p=top_p)
    raise ValueError(
        f"Unknown method '{method_name}'. "
        f"Choose: greedy | bon | bon_majority | greedy_pass_n"
    )
