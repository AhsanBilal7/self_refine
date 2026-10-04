"""Process-reward-model (PRM) scoring of reasoning traces.

The breadth-depth refinement method itself is verifier-free. A PRM is used
for

* the ``bon`` (Best-of-N with verifier) baseline in ``run_eval.py``, and
* diagnostic scores stored with every refinement step (``*_prm_score``); these
  only break exact vote ties and never change which trace is refined.

``HFProcessRewardVerifier`` implements the Qwen2.5-Math-PRM-7B protocol: the
response is split into steps joined by the ``<extra_0>`` separator, the full
(system, user, assistant) conversation is scored in one forward pass, and the
positive-class probability at each separator is that step's reward. Step
rewards are aggregated per trace by ``min`` (default), ``mean`` or ``last``.
"""

import logging
import re
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Step separator token used by Qwen2.5-Math-PRM-7B
STEP_SEP = "<extra_0>"

# System prompt used during PRM training — must match at scoring time
PRM_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: extract the boxed answer from a model trace
# ──────────────────────────────────────────────────────────────────────────────

def extract_boxed_answer(trace: str) -> str:
    """Return the content of the last \\boxed{} in a trace, or ''."""
    if "boxed" not in trace:
        return ""
    ans = trace.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack, a = 1, ""
        for c in ans[1:]:
            if c == "{":
                stack += 1; a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
        return a
    return ans.split("$")[0].strip()


# ──────────────────────────────────────────────────────────────────────────────
# Process-reward verifier (Qwen2.5-Math-PRM-7B)
# ──────────────────────────────────────────────────────────────────────────────

class HFProcessRewardVerifier:
    """Score traces with a Qwen2.5-Math-PRM-style process reward model.

    The model is loaded with ``AutoModel``; its output at each ``<extra_0>``
    position holds two logits whose softmax gives the step reward. Traces are
    scored one at a time so each step sees its full left context.

    Parameters
    ----------
    model_name    : PRM checkpoint (default ``Qwen/Qwen2.5-Math-PRM-7B``).
    aggregation   : ``min`` | ``mean`` | ``last`` over step rewards.
    device        : ``auto`` (device_map) or an explicit device string.
    cache_dir     : Hugging Face cache directory.
    batch_size    : kept for interface compatibility; scoring is per trace.
    system_prompt : system message used at scoring time (PRM training format).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        aggregation: str = "min",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        batch_size: int = 8,
        system_prompt: str = PRM_SYSTEM_PROMPT,
    ):
        self.aggregation   = aggregation
        self.batch_size    = batch_size
        self.system_prompt = system_prompt

        import torch
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading PRM tokenizer: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        # Rewards are read at <extra_0> positions, so padding side only
        # affects the attention mask; right padding keeps real tokens
        # contiguous.
        self._tokenizer.padding_side = "right"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        # Cache the <extra_0> token id once — used in every forward pass
        self._step_sep_id = self._tokenizer.encode(STEP_SEP)[0]
        logger.info(f"  step_sep_id ({STEP_SEP!r}) = {self._step_sep_id}")

        logger.info(f"Loading PRM model: {model_name}")
        self._model = AutoModel.from_pretrained(
            model_name,
            device_map=device if device == "auto" else None,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self._model.config.use_cache = False
        if device not in ("auto",):
            self._model = self._model.to(device)
        self._model.eval()
        self._device = next(self._model.parameters()).device
        logger.info(
            f"PRM loaded on {self._device}  |  "
            f"aggregation={aggregation}  batch_size={batch_size}"
        )

    # ── step splitting ────────────────────────────────────────────────────────

    def _split_steps(self, trace: str) -> List[str]:
        """
        Split a model trace into reasoning steps.

        Strategy:
          1. If the trace already contains the PRM step separator, split on it.
          2. Otherwise split on double newlines or 'Step N:' markers.
          3. Always return at least one step (the full trace).
        """
        if STEP_SEP in trace:
            steps = [s.strip() for s in trace.split(STEP_SEP) if s.strip()]
            return steps if steps else [trace]

        steps = [
            s.strip()
            for s in re.split(r"\n\n+|Step \d+[:.]\s*", trace)
            if s.strip()
        ]
        return steps if steps else [trace]

    # ── chat template builder ─────────────────────────────────────────────────

    def _build_conversation_str(self, question: str, steps: List[str]) -> str:
        """
        Build the full chat-formatted string for one trace.

        Format (matches PRM training format):
          [system prompt]
          [user: question]
          [assistant: step1<extra_0>step2<extra_0>...stepN<extra_0>]

        The <extra_0> token is appended after EVERY step including the last.
        """
        assistant_content = STEP_SEP.join(steps) + STEP_SEP
        messages = [
            {"role": "system",    "content": self.system_prompt},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": assistant_content},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # ── per-step reward extraction ────────────────────────────────────────────

    @staticmethod
    def _make_step_rewards(logits, token_masks) -> List[List[float]]:
        """
        Extract per-step reward scores from the model's hidden-state logits.

        Follows the Qwen2.5-Math-PRM-7B reference usage.

        Parameters
        ----------
        logits      : tensor (batch, seq_len, 2)  — model output hidden states
        token_masks : tensor (batch, seq_len)      — True at <extra_0> positions

        Returns
        -------
        List[List[float]] — one list of step scores per sequence in the batch
        """
        import torch.nn.functional as F
        probabilities = F.softmax(logits, dim=-1)                       # (B, L, 2)
        probabilities = probabilities * token_masks.unsqueeze(-1)       # zero non-sep positions
        all_scores = []
        for i in range(probabilities.size(0)):
            sample        = probabilities[i]                            # (L, 2)
            # Extract rows where token_mask was True (non-zero rows)
            positive_probs = sample[sample.sum(-1) != 0][:, 1]         # (n_steps,)
            all_scores.append(positive_probs.cpu().tolist())
        return all_scores

    # ── single-trace scoring ──────────────────────────────────────────────────

    def _score_single_trace(self, question: str, trace: str) -> float:
        """
        Score one trace by running a single forward pass over the full
        conversation sequence.  Returns aggregated step reward (float).
        """
        import torch

        steps   = self._split_steps(trace)
        conv    = self._build_conversation_str(question, steps)
        input_ids = self._tokenizer.encode(
            conv, return_tensors="pt"
        ).to(self._device)

        token_masks = (input_ids == self._step_sep_id)   # (1, seq_len)

        if token_masks.sum() == 0:
            # No step separators found — degenerate trace; return 0.0
            logger.debug("No <extra_0> tokens found in trace; returning 0.0")
            return 0.0

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids)

        # outputs[0] is the last hidden state: (1, seq_len, hidden_dim)
        # For Qwen2.5-Math-PRM-7B, the model head maps hidden_dim → 2 labels
        logits = outputs[0]                                   # (1, seq_len, 2)
        step_scores = self._make_step_rewards(logits, token_masks)[0]

        if not step_scores:
            return 0.0

        if self.aggregation == "min":
            return float(min(step_scores))
        elif self.aggregation == "last":
            return float(step_scores[-1])
        else:  # mean
            return float(np.mean(step_scores))

    # ── batched public API ────────────────────────────────────────────────────

    def score_batch(
        self,
        traces: List[str],
        question: Optional[str] = None,
    ) -> np.ndarray:
        """
        Score a list of candidate traces for ONE question.

        Parameters
        ----------
        traces   : N candidate traces (all answers to the same question)
        question : the question text, needed for the PRM input format. If
                   None, traces are scored with an empty user turn.

        Returns
        -------
        np.ndarray of float scores, shape (N,), higher = better reasoning.
        """
        if question is None:
            logger.warning(
                "score_batch called without question context. "
                "PRM scores will be less accurate. "
                "Pass question= for correct results."
            )
            question = ""

        scores = []
        for trace in traces:
            scores.append(self._score_single_trace(question, trace))
        return np.array(scores, dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_verifier(
    verifier_type: str = "prm",
    data_name: str = "math",
    prm_model: Optional[str] = None,
    prm_aggregation: str = "min",
    device: str = "auto",
    cache_dir: Optional[str] = None,
    batch_size: int = 8,
):
    """Build the PRM verifier (``verifier_type`` must be ``"prm"``).

    ``data_name`` is accepted for interface compatibility and is unused.
    """
    if verifier_type != "prm":
        raise ValueError(f"Unknown verifier_type: {verifier_type!r} (expected 'prm')")
    return HFProcessRewardVerifier(
        model_name=prm_model or "Qwen/Qwen2.5-Math-PRM-7B",
        aggregation=prm_aggregation,
        device=device,
        cache_dir=cache_dir,
        batch_size=batch_size,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Unified scoring entry point
# ──────────────────────────────────────────────────────────────────────────────

def score_candidates(
    traces: List[str],
    verifier,
    gold_answer: Optional[str] = None,
    question: Optional[str] = None,
) -> np.ndarray:
    """Score the candidate traces for one question (higher is better).

    ``gold_answer`` is accepted for interface compatibility and ignored; the
    verifier never sees the gold answer.
    """
    if isinstance(verifier, HFProcessRewardVerifier):
        return verifier.score_batch(traces, question=question)
    raise TypeError(f"Unsupported verifier type: {type(verifier)}")
