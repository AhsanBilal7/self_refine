"""N rollouts x D refinement depths for one problem (Algorithm 1, lines 1-9).

Depths are processed sequentially because depth ``d`` needs ``r_i^(d-1)``.
Within a depth, every rollout that still needs that depth is sent through
``run_step_batch`` together, so each stage (continuation, critique,
correction) is one batched generation call across the ``N`` rollouts::

    d = 1:  sample r^(0) for all N rollouts -> critique -> correct -> r^(1)
    d = 2:  continue from r^(1)             -> critique -> correct -> r^(2)
    ...
    d = D:  ...                                                    -> r^(D)

Resume. ``resume_states`` holds one previously saved rollout record (or
``None``) per rollout. A rollout with ``k`` saved steps re-enters at depth
``k + 1``; saved steps beyond the requested ``D`` are dropped.
``on_rollout_step_complete`` fires after every (rollout, depth) step so the
caller can persist progress.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .cimr_step import CIMRStepConfig, run_step_batch

logger = logging.getLogger(__name__)


def _error_step_record(
    rollout_idx: int,
    step_idx: int,
    total_steps: int,
    prior: Optional[str],
    error: str = "",
) -> Dict[str, Any]:
    """Placeholder step for a failed batch; carries ``r^(d-1)`` forward."""
    return dict(
        rollout_idx=int(rollout_idx),
        step_idx=int(step_idx),
        total_steps=int(total_steps),
        reasoning_response="",
        reasoning_answer="",
        reasoning_prm_score=0.0,
        critique_response="",
        correction_response="",
        corrected_trace=prior or "",
        corrected_answer="",
        corrected_prm_score=0.0,
        elapsed_sec=0.0,
        error=error,
    )


def _build_rollout_record(
    *,
    rollout_idx: int,
    total_steps: int,
    steps: List[Dict[str, Any]],
    completed: bool,
    elapsed_sec: float,
) -> Dict[str, Any]:
    """Summarise a rollout from its step records.

    ``final_answer`` is ``Extract(r_i^(D))``, the rollout's vote ``a_i``.
    """
    if steps:
        last = steps[-1]
        final_answer = last.get("corrected_answer", "") or last.get("reasoning_answer", "")
        final_prm_score = float(last.get("corrected_prm_score", 0.0))
        final_corrected = last.get("corrected_trace", "")
    else:
        final_answer = ""
        final_prm_score = 0.0
        final_corrected = ""

    return dict(
        rollout_idx=int(rollout_idx),
        total_steps=int(total_steps),
        completed=bool(completed),
        n_steps_done=len(steps),
        steps=list(steps),
        final_answer=final_answer,
        final_prm_score=final_prm_score,
        final_corrected_trace=final_corrected,
        elapsed_sec=round(float(elapsed_sec), 3),
    )


def run_all_rollouts_batched(
    *,
    model,
    verifier,
    question: str,
    gold_answer: Optional[str],
    n_rollouts: int,
    total_steps: int,
    config: CIMRStepConfig,
    on_rollout_step_complete: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    resume_states: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Run ``n_rollouts`` rollouts for ``total_steps`` refinement depths.

    Parameters
    ----------
    n_rollouts               : breadth ``N``.
    total_steps              : depth ``D``.
    on_rollout_step_complete : ``callback(rollout_idx, running_record)``.
    resume_states            : length-``N`` list of saved rollout records or
                               ``None`` entries.

    Returns
    -------
    ``N`` rollout records (see ``_build_rollout_record``).
    """
    t0 = time.time()

    all_steps: List[List[Dict[str, Any]]] = [[] for _ in range(n_rollouts)]
    if resume_states:
        for i in range(n_rollouts):
            rs = resume_states[i] if i < len(resume_states) else None
            if rs and isinstance(rs, dict):
                steps = sorted(rs.get("steps", []), key=lambda s: int(s.get("step_idx", 0)))
                all_steps[i] = [s for s in steps if int(s.get("step_idx", 0)) <= total_steps]

    n_done = [len(all_steps[i]) for i in range(n_rollouts)]
    prior_corrected: List[Optional[str]] = [
        (all_steps[i][-1].get("corrected_trace") if all_steps[i] else None)
        for i in range(n_rollouts)
    ]

    for step_idx in range(1, total_steps + 1):
        to_run = [i for i in range(n_rollouts) if n_done[i] < step_idx]
        if not to_run:
            logger.debug(
                f"[CIMR batch] depth {step_idx}: "
                f"all {n_rollouts} rollouts already complete."
            )
            continue

        logger.info(
            f"[CIMR batch] depth {step_idx}/{total_steps}  "
            f"batch_size={len(to_run)}  rollout_idxs={to_run}"
        )

        try:
            step_records = run_step_batch(
                model=model,
                verifier=verifier,
                question=question,
                gold_answer=gold_answer,
                step_idx=step_idx,
                total_steps=total_steps,
                prior_corrected_list=[prior_corrected[i] for i in to_run],
                config=config,
                rollout_idxs=to_run,
            )
        except Exception as e:
            logger.error(f"[CIMR batch] depth {step_idx} batch failed: {e}", exc_info=True)
            step_records = [
                _error_step_record(
                    rollout_idx=i,
                    step_idx=step_idx,
                    total_steps=total_steps,
                    prior=prior_corrected[i],
                    error=str(e),
                )
                for i in to_run
            ]

        for record, rollout_i in zip(step_records, to_run):
            all_steps[rollout_i].append(record)
            n_done[rollout_i] = len(all_steps[rollout_i])
            prior_corrected[rollout_i] = (
                record.get("corrected_trace") or prior_corrected[rollout_i]
            )

            if on_rollout_step_complete is not None:
                running = _build_rollout_record(
                    rollout_idx=rollout_i,
                    total_steps=total_steps,
                    steps=all_steps[rollout_i],
                    completed=(n_done[rollout_i] >= total_steps),
                    elapsed_sec=time.time() - t0,
                )
                try:
                    on_rollout_step_complete(rollout_i, running)
                except Exception as cb_err:
                    logger.warning(
                        f"[CIMR batch] on_rollout_step_complete failed "
                        f"(rollout {rollout_i}, depth {step_idx}): {cb_err}"
                    )

    total_elapsed = time.time() - t0
    return [
        _build_rollout_record(
            rollout_idx=i,
            total_steps=total_steps,
            steps=all_steps[i],
            completed=True,
            elapsed_sec=total_elapsed,
        )
        for i in range(n_rollouts)
    ]


__all__ = ["run_all_rollouts_batched"]
