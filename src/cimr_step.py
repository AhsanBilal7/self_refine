"""One refinement depth of breadth-depth refinement, batched over rollouts.

For every rollout ``i`` processed at depth ``d`` this module runs the
three-stage cycle of the paper (Section "Iterative Self-Critique and
Self-Correction"):

1. Reasoning continuation  ``g_i^(d) ~ pi^G(. | x, r_i^(d-1))``
2. Self-critique           ``c_i^(d) ~ pi^C(. | x, g_i^(d))``
3. Self-correction         ``r_i^(d) ~ pi^R(. | x, g_i^(d), c_i^(d))``

and returns one step record per rollout. ``r_i^(d)`` is fed to depth
``d + 1``; after depth ``D`` it is the terminal trace used for voting.

Depth indexing. At ``d = 1`` there is no previous corrected trace, so the
generator samples from the problem alone at the rollout temperature. That
sample is the initial rollout ``r_i^(0)`` and is critiqued and corrected
directly, i.e. it also plays the role of ``g_i^(1)``. At ``d >= 2`` the
generator continues from ``r_i^(d-1)``. A run with depth ``D`` therefore makes
``3 * D`` generation calls per rollout.

Decoding. Only the ``d = 1`` generation is sampled (temperature / top-p from
``CIMRStepConfig``), which is where breadth diversity comes from. Reasoning
continuation (``d >= 2``), critique and correction are decoded greedily.

Process-reward scores. When a verifier is supplied, ``g_i^(d)`` and
``r_i^(d)`` are scored with a PRM for diagnostics only. The scores never
influence the trace that is carried forward.

Step-record fields (stored in ``cimr_full_results_*.json``)::

    reasoning_response   g_i^(d), the full generator output
    reasoning_answer     Extract(g_i^(d))              (alpha in the appendix)
    critique_response    c_i^(d)  ("" under --no_critique)
    correction_response  the full corrector output
    corrected_trace      r_i^(d)  (falls back to g_i^(d) if the corrector
                                   returns nothing or --no_correction is set)
    corrected_answer     Extract(r_i^(d))              (beta in the appendix)
    reasoning_prm_score / corrected_prm_score   diagnostic PRM scores
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .cimr_prompts import (
    CORRECTION_SYSTEM_PROMPT,
    CRITIQUE_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    build_continuation_reasoning_prompt,
    build_initial_reasoning_prompt,
    build_self_correction_prompt,
    build_self_critique_prompt,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Answer extraction
# =============================================================================

def _last_number(text: str) -> str:
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", (text or "").replace(",", ""))
    return nums[-1] if nums else ""


def extract_boxed(text: str) -> str:
    """Return the content of the last ``\\boxed{...}`` (nested braces allowed).

    Falls back to the last number in ``text`` when no ``\\boxed{`` is present.
    """
    if not text:
        return ""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return _last_number(text)
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start: i - 1].strip() if depth == 0 else text[start:].strip()


def extract_answer(text: str) -> str:
    """Deterministic answer extractor ``Extract(.)`` used for every trace.

    The last ``\\boxed{}`` content, or the last number in the text when the
    box is missing or empty.
    """
    return extract_boxed(text) or _last_number(text)


# =============================================================================
# Prompt formatting and batched generation
# =============================================================================

def build_chat_prompt(model, system_content: str, user_content: str) -> str:
    """Format a (system, user) turn with the model's chat template.

    Falls back to ``"<system>\\n\\n<user>"`` when the tokenizer has no chat
    template.
    """
    tok = getattr(model, "tokenizer", None)
    if tok is not None and getattr(tok, "chat_template", None):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ]
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(f"apply_chat_template failed ({e}); falling back.")
    return f"{system_content}\n\n{user_content}"


def _generate_batch(
    model,
    prompts: List[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> List[str]:
    """Generate one completion per prompt in chunks of ``model.batch_size``.

    A chunk that raises is logged and returns empty strings, so one failure
    does not abort the other rollouts.
    """
    if not prompts:
        return []

    kwargs: Dict[str, Any] = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if do_sample:
        kwargs.update(temperature=temperature, top_p=top_p)

    bs = getattr(model, "batch_size", len(prompts)) or len(prompts)
    results: List[str] = []
    for start in range(0, len(prompts), bs):
        chunk = prompts[start: start + bs]
        try:
            out = model._generate_batch_raw(chunk, **kwargs)
            results.extend(out if out else [""] * len(chunk))
        except Exception as e:
            logger.error(f"[CIMR] batch generation failed: {e}")
            results.extend([""] * len(chunk))
    return results


# =============================================================================
# Diagnostic PRM scoring
# =============================================================================

def score_traces_batch(
    verifier,
    question: str,
    traces: List[str],
    gold_answer: Optional[str] = None,
) -> List[float]:
    """PRM-score traces for diagnostics; 0.0 for empty traces or no verifier."""
    if not traces:
        return []
    if verifier is None:
        return [0.0] * len(traces)

    valid_idxs = [i for i, t in enumerate(traces) if t and t.strip()]
    scores = [0.0] * len(traces)
    if not valid_idxs:
        return scores

    try:
        from .verifier import score_candidates
        raw = score_candidates(
            [traces[i] for i in valid_idxs], verifier, gold_answer, question=question,
        )
        flat = np.asarray(raw, dtype=float).reshape(-1).tolist()
        for out_i, in_i in enumerate(valid_idxs):
            scores[in_i] = float(flat[out_i]) if out_i < len(flat) else 0.0
    except Exception as e:
        logger.warning(f"[CIMR] batch PRM scoring failed: {e}")
    return scores


# =============================================================================
# Step configuration
# =============================================================================

@dataclass
class CIMRStepConfig:
    """Decoding budgets and ablation switches for one refinement depth.

    Defaults match the values used for the paper's runs
    (``run_paired_ablation.sh``). ``*_chars`` limits truncate traces inside
    prompts (head + tail kept) before they reach the model.
    """
    # Generator (initial rollout at d = 1, continuation at d >= 2)
    reasoning_max_new_tokens:   int   = 2048
    reasoning_temperature:      float = 0.7    # rollout temperature tau (d = 1)
    reasoning_top_p:            float = 0.95
    reasoning_do_sample:        bool  = True   # sample at d = 1; d >= 2 is greedy

    # Critic
    critique_max_tokens:        int   = 384
    critique_max_trace_chars:   int   = 4000

    # Corrector
    correction_max_tokens:         int = 1024
    correction_max_trace_chars:    int = 4000
    correction_max_critique_chars: int = 1500

    # Budget for r^(d-1) inside the continuation prompt
    continuation_max_prior_chars: int = 4000

    # Ablations
    no_critique:   bool = False   # corrector sees an empty critique
    no_correction: bool = False   # r^(d) := g^(d)


# =============================================================================
# One depth, batched over rollouts
# =============================================================================

def run_step_batch(
    *,
    model,
    verifier,
    question: str,
    gold_answer: Optional[str],
    step_idx: int,
    total_steps: int,
    prior_corrected_list: List[Optional[str]],
    config: CIMRStepConfig,
    rollout_idxs: List[int],
) -> List[Dict[str, Any]]:
    """Run refinement depth ``step_idx`` for several rollouts at once.

    Parameters
    ----------
    prior_corrected_list : ``r_i^(d-1)`` for each rollout (``None`` at d = 1).
    rollout_idxs         : rollout index ``i`` for each entry above.

    Returns
    -------
    One step record per rollout, in the order of ``rollout_idxs``.
    """
    n = len(rollout_idxs)
    is_first = (step_idx == 1)
    t0 = time.time()

    logger.debug(
        f"[CIMR step] depth={step_idx}/{total_steps}  "
        f"is_first={is_first}  batch_size={n}"
    )

    # ── 1. Reasoning continuation g^(d) (initial rollout at d = 1) ──────────
    if is_first:
        user_msgs = [build_initial_reasoning_prompt(question)] * n
    else:
        user_msgs = [
            build_continuation_reasoning_prompt(
                question=question,
                prior_corrected=prior_corrected_list[k] or "",
                step_idx=step_idx,
                total_steps=total_steps,
                max_prior_chars=config.continuation_max_prior_chars,
            )
            for k in range(n)
        ]
    reasoning_prompts = [
        build_chat_prompt(model, REASONING_SYSTEM_PROMPT, u) for u in user_msgs
    ]
    reasoning_raws = _generate_batch(
        model, reasoning_prompts,
        max_new_tokens=config.reasoning_max_new_tokens,
        do_sample=config.reasoning_do_sample and is_first,
        temperature=config.reasoning_temperature,
        top_p=config.reasoning_top_p,
    )
    reasoning_cleans = [r or "" for r in reasoning_raws]
    reasoning_answers = [extract_answer(r) for r in reasoning_cleans]
    reasoning_prms = score_traces_batch(verifier, question, reasoning_cleans, gold_answer)

    logger.debug(f"[CIMR step] depth={step_idx} reasoning answers={reasoning_answers}")

    # ── 2. Self-critique c^(d) ───────────────────────────────────────────────
    if config.no_critique:
        critique_raws = [""] * n
    else:
        critique_prompts = [
            build_chat_prompt(
                model, CRITIQUE_SYSTEM_PROMPT,
                build_self_critique_prompt(
                    question=question,
                    trace_text=reasoning_cleans[k],
                    trace_answer=reasoning_answers[k],
                    max_trace_chars=config.critique_max_trace_chars,
                ),
            )
            for k in range(n)
        ]
        critique_raws = _generate_batch(
            model, critique_prompts,
            max_new_tokens=config.critique_max_tokens,
            do_sample=False,
        )

    # ── 3. Self-correction r^(d) ─────────────────────────────────────────────
    if config.no_correction:
        correction_cleans = [""] * n
    else:
        correction_prompts = [
            build_chat_prompt(
                model, CORRECTION_SYSTEM_PROMPT,
                build_self_correction_prompt(
                    question=question,
                    trace_text=reasoning_cleans[k],
                    critique_text=critique_raws[k] or "",
                    trace_answer=reasoning_answers[k],
                    max_trace_chars=config.correction_max_trace_chars,
                    max_critique_chars=config.correction_max_critique_chars,
                ),
            )
            for k in range(n)
        ]
        correction_raws = _generate_batch(
            model, correction_prompts,
            max_new_tokens=config.correction_max_tokens,
            do_sample=False,
        )
        correction_cleans = [r or "" for r in correction_raws]

    # The corrector output becomes r^(d) when non-empty; otherwise g^(d) is
    # carried forward so the next depth never starts from a blank trace.
    corrected_traces: List[str] = []
    corrected_answers: List[str] = []
    for k in range(n):
        corr = correction_cleans[k].strip()
        if corr:
            corrected_traces.append(corr)
            corrected_answers.append(extract_answer(corr) or reasoning_answers[k])
        else:
            corrected_traces.append(reasoning_cleans[k])
            corrected_answers.append(reasoning_answers[k])

    logger.debug(f"[CIMR step] depth={step_idx} corrected answers={corrected_answers}")

    corrected_prms = score_traces_batch(verifier, question, corrected_traces, gold_answer)

    per_rollout_time = round((time.time() - t0) / max(n, 1), 3)
    return [
        dict(
            rollout_idx=int(rollout_idxs[k]),
            step_idx=int(step_idx),
            total_steps=int(total_steps),
            reasoning_response=reasoning_cleans[k],
            reasoning_answer=reasoning_answers[k],
            reasoning_prm_score=float(reasoning_prms[k]),
            critique_response=critique_raws[k],
            correction_response=correction_cleans[k],
            corrected_trace=corrected_traces[k],
            corrected_answer=corrected_answers[k],
            corrected_prm_score=float(corrected_prms[k]),
            elapsed_sec=per_rollout_time,
        )
        for k in range(n)
    ]


__all__ = [
    "CIMRStepConfig",
    "run_step_batch",
    "extract_answer",
    "extract_boxed",
    "build_chat_prompt",
    "score_traces_batch",
]
