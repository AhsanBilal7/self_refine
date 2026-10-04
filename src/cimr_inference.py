"""Breadth-depth refinement orchestrator ("CIMR" in code).

CIMR is the code name for the paper's training-free, verifier-free
breadth-depth framework (Algorithm 1). Per problem ``x`` it

1. samples ``N`` independent rollouts (breadth),
2. refines each through ``D`` rounds of reasoning continuation, self-critique
   and self-correction (depth), see ``cimr_step``, and
3. extracts one answer per terminal trace and takes a plurality vote, see
   ``cimr_aggregator``.

Problems run one after another; within a problem all ``N`` rollouts are
batched at every depth.

Persistence. All state lives in one JSON store per run,
``<output_dir>/cimr_full_results_<dataset>_N<N>_d<D><suffix>.json``, keyed by
question id and rewritten atomically after every (rollout, depth) step.
Re-running the same command skips completed problems and resumes partially
completed rollouts from their last saved depth.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .cimr_aggregator import aggregate_rollouts
from .cimr_rollout import run_all_rollouts_batched
from .cimr_step import CIMRStepConfig

logger = logging.getLogger(__name__)


@dataclass
class CIMRConfig:
    """Run-level settings.

    n_rollouts  : breadth ``N``, independent rollouts per problem.
    depth       : depth ``D``, refinement rounds per rollout.
    step_config : per-depth decoding budgets and ablation switches.
    tag_suffix  : appended to the store filename (e.g. ``_no_critique``).
    """
    n_rollouts: int = 8
    depth: int = 4
    step_config: CIMRStepConfig = field(default_factory=CIMRStepConfig)
    tag_suffix: str = ""


# =============================================================================
# Results store
# =============================================================================

def cimr_store_path(output_dir: str, dataset: str, N: int, D: int, suffix: str = "") -> str:
    """Path of the per-run results store."""
    return os.path.join(
        output_dir, f"cimr_full_results_{dataset}_N{N}_d{D}{suffix}.json"
    )


def open_cimr_store(
    output_dir: str, dataset: str, N: int, D: int, suffix: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Load the results store if it exists, else start an empty one.

    Returns ``(path, store)`` where ``store`` maps question id to its record.
    """
    path = cimr_store_path(output_dir, dataset, N, D, suffix)
    store: Dict[str, Any] = {}

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                store = data
            elif isinstance(data, list):
                store = {str(r.get("id", i)): r for i, r in enumerate(data)}
            logger.info(f"[CIMR] Resumed {len(store)} questions from {path}")
        except Exception as e:
            logger.warning(f"[CIMR] Could not load store {path}: {e}")
    else:
        logger.info(f"[CIMR] Starting fresh store at {path}")

    return path, store


def save_cimr_store(path: str, store: Dict[str, Any]) -> None:
    """Write the store atomically (temp file, then rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _empty_question_record(
    qid: str,
    question: str,
    gold: Optional[str],
    N: int,
    D: int,
) -> Dict[str, Any]:
    return dict(
        id=qid,
        question=question,
        gold_answer=gold,
        N=N,
        depth=D,
        rollouts=[None] * N,
        completed=False,
        aggregation=None,
        elapsed_sec=0.0,
    )


def _question_is_complete(qrec: Dict[str, Any]) -> bool:
    """True iff all ``N`` rollouts exist and have finished all ``D`` depths."""
    rollouts = qrec.get("rollouts") or []
    if len(rollouts) < int(qrec.get("N", 0)):
        return False
    D = int(qrec.get("depth", 0))
    for r in rollouts:
        if r is None:
            return False
        if int(r.get("n_steps_done", 0)) < D:
            return False
        if not r.get("completed", False):
            return False
    return True


# =============================================================================
# Orchestrator
# =============================================================================

class CIMRInferencer:
    """Runs breadth-depth refinement problem by problem with resumable state.

    Parameters
    ----------
    model      : generator wrapper exposing ``tokenizer``, ``batch_size`` and
                 ``_generate_batch_raw`` (see ``model_wrapper.HFModelWrapper``).
    verifier   : optional PRM used only for diagnostic scores; ``None`` skips it.
    ev         : dataset evaluator for correctness checks against the gold answer.
    data_name  : canonical dataset name for the evaluator.
    config     : ``CIMRConfig``.
    output_dir : directory of the results store.
    dataset    : dataset tag used in the store filename.
    """

    def __init__(
        self,
        model,
        verifier,
        ev,
        data_name: str,
        config: CIMRConfig,
        output_dir: str,
        dataset: str,
    ):
        self.model = model
        self.verifier = verifier
        self.ev = ev
        self.data_name = data_name
        self.config = config
        self.output_dir = output_dir
        self.dataset = dataset

        os.makedirs(output_dir, exist_ok=True)
        self.store_path, self.store = open_cimr_store(
            output_dir=output_dir,
            dataset=dataset,
            N=config.n_rollouts,
            D=config.depth,
            suffix=config.tag_suffix,
        )

    def run_question(
        self,
        question: str,
        gold_answer: Optional[str],
        qid: str,
    ) -> Dict[str, Any]:
        """Refine and aggregate one problem, reusing any saved progress.

        Returns the question record stored under ``qid``.
        """
        N = int(self.config.n_rollouts)
        D = int(self.config.depth)
        qid = str(qid)

        qrec = self.store.get(qid)
        if qrec is None:
            qrec = _empty_question_record(qid, question, gold_answer, N, D)
            self.store[qid] = qrec
            save_cimr_store(self.store_path, self.store)
        else:
            # Pad the rollout list if N increased across runs.
            existing = list(qrec.get("rollouts", []))
            if len(existing) < N:
                qrec["rollouts"] = existing + [None] * (N - len(existing))

        if _question_is_complete(qrec):
            logger.debug(f"[CIMR q={qid}] already complete ({N}x{D}); skipping.")
            if qrec.get("aggregation") is None:
                qrec["aggregation"] = aggregate_rollouts(
                    rollouts=qrec.get("rollouts", []),
                    gold_answer=qrec.get("gold_answer"),
                    ev=self.ev,
                    data_name=self.data_name,
                )
                qrec["completed"] = True
                save_cimr_store(self.store_path, self.store)
            return qrec

        t_q = time.time()
        elapsed_baseline = float(qrec.get("elapsed_sec", 0.0))

        def on_rollout_step_complete(rollout_i: int, running_record: Dict[str, Any]) -> None:
            qrec["rollouts"][rollout_i] = running_record
            qrec["elapsed_sec"] = round(elapsed_baseline + (time.time() - t_q), 3)
            save_cimr_store(self.store_path, self.store)

        qrec["rollouts"] = run_all_rollouts_batched(
            model=self.model,
            verifier=self.verifier,
            question=question,
            gold_answer=gold_answer,
            n_rollouts=N,
            total_steps=D,
            config=self.config.step_config,
            on_rollout_step_complete=on_rollout_step_complete,
            resume_states=[qrec["rollouts"][i] for i in range(N)],
        )
        qrec["aggregation"] = aggregate_rollouts(
            rollouts=qrec["rollouts"],
            gold_answer=gold_answer,
            ev=self.ev,
            data_name=self.data_name,
        )
        qrec["completed"] = True
        qrec["elapsed_sec"] = round(elapsed_baseline + (time.time() - t_q), 3)
        save_cimr_store(self.store_path, self.store)
        return qrec


__all__ = [
    "CIMRConfig",
    "CIMRInferencer",
    "cimr_store_path",
    "open_cimr_store",
    "save_cimr_store",
]
