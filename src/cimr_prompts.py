"""Role prompts for breadth-depth refinement (paper Appendix "Role Prompts").

One base model plays three roles that differ only in their prompt:

* generator ``pi^G`` -- the initial rollout ``r_i^(0)`` and the reasoning
  continuation ``g_i^(d) ~ pi^G(. | x, r_i^(d-1))`` (Eq. reasoning);
* critic ``pi^C`` -- the critique ``c_i^(d) ~ pi^C(. | x, g_i^(d))``
  (Eq. critique);
* corrector ``pi^R`` -- the corrected trace
  ``r_i^(d) ~ pi^R(. | x, g_i^(d), c_i^(d))`` (Eq. correction).

All prompts are zero-shot and built only from the problem and the traces of
the current rollout. The prompt text is reproduced verbatim in the paper, so
edits here must be mirrored there.
"""

from __future__ import annotations

import re


def _truncate_middle(text: str, max_chars: int) -> str:
    """Keep the head and tail of ``text`` when it exceeds ``max_chars``."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    head = text[:keep].rstrip()
    tail = text[-keep:].lstrip()
    omitted = len(text) - max_chars
    return f"{head}\n\n[... {omitted:,} characters omitted ...]\n\n{tail}"


# ---------------------------------------------------------------------------
# System prompts (one per role)
# ---------------------------------------------------------------------------

REASONING_SYSTEM_PROMPT = (
    "You are a precise mathematical reasoner.\n"
    "Solve problems using clearly numbered steps.\n"
    "Show every calculation inside each step.\n"
    "End your response with exactly one line:\n"
    "Final answer: \\boxed{<answer>}"
)

CRITIQUE_SYSTEM_PROMPT = (
    "You are a careful mathematical critic.\n"
    "You will be given a PROBLEM and a SOLUTION TRACE.\n"
    "Your job: identify any errors in the trace — logical mistakes, "
    "wrong calculations, incorrect assumptions, or faulty reasoning.\n"
    "If the trace is correct, say so explicitly.\n"
    "Be concise and specific. Do NOT solve the problem yourself."
)

CORRECTION_SYSTEM_PROMPT = (
    "You are a careful mathematical editor.\n"
    "You will be given a PROBLEM, a SOLUTION TRACE, and a CRITIQUE.\n"
    "Your job: produce a corrected solution that fixes all errors identified "
    "in the critique while keeping the correct parts of the original trace.\n"
    "Use the same numbered-step format.\n"
    "End your response with exactly one line:\n"
    "Final answer: \\boxed{<answer>}"
)


# ---------------------------------------------------------------------------
# Generator: initial rollout and reasoning continuation
# ---------------------------------------------------------------------------

def build_initial_reasoning_prompt(question: str) -> str:
    """User prompt for the initial rollout, conditioned on the problem only."""
    q = (question or "").strip()
    return (
        "Solve the following problem step by step.\n\n"
        "PROBLEM\n"
        f"{q}\n\n"
        "Use numbered steps (Step 1:, Step 2:, ...). "
        "Show every calculation. "
        "End with exactly one line:\n"
        "Final answer: \\boxed{<answer>}"
    )


def build_continuation_reasoning_prompt(
    question: str,
    prior_corrected: str,
    step_idx: int,
    total_steps: int,
    max_prior_chars: int = 3000,
) -> str:
    """User prompt for reasoning continuation ``g^(d) ~ pi^G(. | x, r^(d-1))``.

    Parameters
    ----------
    question        : the input problem ``x``.
    prior_corrected : the corrected trace ``r^(d-1)`` from the previous depth.
    step_idx        : current refinement depth ``d`` (>= 2).
    total_steps     : total refinement depth ``D``.
    max_prior_chars : character budget for ``r^(d-1)`` (head + tail kept).
    """
    q = (question or "").strip()
    prior = (prior_corrected or "").strip()

    prior = _truncate_middle(prior, max_prior_chars)

    # Continue the step numbering of the prior trace.
    step_nums = re.findall(r"^Step\s+(\d+)\s*:", prior, re.MULTILINE | re.IGNORECASE)
    last_num = int(step_nums[-1]) if step_nums else (step_idx - 1)
    next_num = last_num + 1

    is_final = (step_idx >= total_steps)
    final_note = " This is the FINAL refinement depth." if is_final else ""

    return (
        f"You are at refinement depth {step_idx} of {total_steps}.{final_note}\n\n"
        "Below is the ORIGINAL PROBLEM and the CORRECTED REASONING from the "
        "previous depth.\n\n"
        "Your task:\n"
        f"  1. Review the corrected reasoning carefully.\n"
        f"  2. Continue from Step {next_num}: — extend or refine the solution, "
        "verifying each claim.\n"
        "  3. If the prior reasoning already reaches a correct answer with no "
        "errors, confirm it concisely. Do NOT change the answer unless you can "
        "identify a specific concrete error.\n"
        "  4. End with exactly one line:\n"
        "     Final answer: \\boxed{<answer>}\n\n"
        "PROBLEM\n"
        f"{q}\n\n"
        f"CORRECTED REASONING FROM DEPTH {step_idx - 1}\n"
        f"{prior if prior else '(none)'}\n\n"
        f"Continue from Step {next_num}:, ending with "
        "`Final answer: \\boxed{{<answer>}}`."
    )


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

def build_self_critique_prompt(
    question: str,
    trace_text: str,
    trace_answer: str,
    max_trace_chars: int = 3000,
) -> str:
    """User prompt for self-critique ``c^(d) ~ pi^C(. | x, g^(d))``.

    The critic points to the erroneous step or replies with the fixed
    confirmation phrase ``The solution appears correct.``

    Parameters
    ----------
    question        : the input problem ``x``.
    trace_text      : the reasoning trace ``g^(d)`` to critique.
    trace_answer    : the answer extracted from ``g^(d)``.
    max_trace_chars : character budget for ``g^(d)`` (head + tail kept).
    """
    q = (question or "").strip()
    trace = (trace_text or "").strip()

    trace = _truncate_middle(trace, max_trace_chars)

    ans_line = f"Extracted answer: {trace_answer}" if trace_answer else "Extracted answer: (none found)"

    return (
        "Review the following SOLUTION TRACE for the PROBLEM below.\n"
        "Identify any errors: wrong calculations, incorrect assumptions, "
        "logical mistakes, or missing steps.\n"
        "If no errors exist, state: 'The solution appears correct.'\n"
        "Be specific — point to the exact step where an error occurs.\n"
        "Do NOT re-solve the problem.\n\n"
        "PROBLEM\n"
        f"{q}\n\n"
        "SOLUTION TRACE\n"
        f"{trace}\n\n"
        f"{ans_line}\n\n"
        "CRITIQUE (list errors specifically, or confirm correctness):"
    )


# ---------------------------------------------------------------------------
# Corrector
# ---------------------------------------------------------------------------

def build_self_correction_prompt(
    question: str,
    trace_text: str,
    critique_text: str,
    trace_answer: str,
    max_trace_chars: int = 3000,
    max_critique_chars: int = 1000,
) -> str:
    """User prompt for self-correction ``r^(d) ~ pi^R(. | x, g^(d), c^(d))``.

    When ``critique_text`` is empty (the ``--no_critique`` ablation), a fixed
    placeholder line is shown in the CRITIQUE slot instead.

    Parameters
    ----------
    question           : the input problem ``x``.
    trace_text         : the reasoning trace ``g^(d)`` to correct.
    critique_text      : the critique ``c^(d)``; empty under ``--no_critique``.
    trace_answer       : the answer extracted from ``g^(d)`` (not shown in the
                         prompt; kept for a uniform builder signature).
    max_trace_chars    : character budget for ``g^(d)`` (head + tail kept).
    max_critique_chars : character budget for ``c^(d)`` (head kept).
    """
    q = (question or "").strip()
    trace = (trace_text or "").strip()
    critique = (critique_text or "").strip()

    trace = _truncate_middle(trace, max_trace_chars)

    if max_critique_chars > 0 and len(critique) > max_critique_chars:
        critique = critique[:max_critique_chars].rstrip() + "\n[critique truncated]"

    return (
        "You are given a PROBLEM, a SOLUTION TRACE, and a CRITIQUE of that trace.\n"
        "Produce a corrected solution:\n"
        "  - Fix every error identified in the critique.\n"
        "  - Keep the numbered-step format (Step 1:, Step 2:, ...).\n"
        "  - Preserve correct parts of the original trace.\n"
        "  - Do NOT introduce new errors.\n"
        "  - End with exactly one line: Final answer: \\boxed{<answer>}\n\n"
        "PROBLEM\n"
        f"{q}\n\n"
        "ORIGINAL SOLUTION TRACE\n"
        f"{trace}\n\n"
        "CRITIQUE\n"
        f"{critique if critique else '(no specific errors identified — solution may be correct)'}\n\n"
        "CORRECTED SOLUTION (numbered steps, ending with Final answer: \\boxed{<answer>}):"
    )


__all__ = [
    "REASONING_SYSTEM_PROMPT",
    "CRITIQUE_SYSTEM_PROMPT",
    "CORRECTION_SYSTEM_PROMPT",
    "build_initial_reasoning_prompt",
    "build_continuation_reasoning_prompt",
    "build_self_critique_prompt",
    "build_self_correction_prompt",
]
