"""Answer extraction and equivalence checking per benchmark family.

``get_evaluator(task)`` returns one of

* ``MATHEvaluator``     -- MATH-500, AIME, AMC (and the ``custom`` loader),
* ``OlympiadEvaluator`` -- OlympiadBench,
* ``GSM8KEvaluator``    -- GSM8K.

``extract_answer_from_model_completion`` reads the last ``\\boxed{}``.
``check_answers_equiv`` tests equivalence symmetrically through a cascade:
exact match, whitespace/``\\left``/``\\right``-normalised match, symbolic
``latex_equiv`` and, for MATH-style answers, numeric closeness.
"""

import re
from typing import Tuple

try:
    from .math_toolkit.latex_answer_check import latex_equiv
except ImportError:  # sympy / pyparsing unavailable
    def latex_equiv(a: str, b: str) -> bool:
        return a.strip() == b.strip()


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX normalisation helper
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_latex(s: str) -> str:
    """
    Normalise a LaTeX string so superficial formatting differences
    do not cause false negatives.

    - Collapse all whitespace runs to one space
    - Remove spaces inside/around braces and parentheses
    - Remove \\left and \\right (decorative)
    - Normalise comma spacing
    """
    # 1. Collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # 2. Spaces around braces
    s = re.sub(r'\{\s+', '{', s)
    s = re.sub(r'\s+\}', '}', s)
    # 3. Spaces around parentheses
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)
    # 4. Remove \left / \right
    s = s.replace('\\left', '').replace('\\right', '')
    # 5. Normalise comma spacing
    s = re.sub(r'\s*,\s*', ',', s)
    # 6. Remove spaces before ] and [ brackets
    s = re.sub(r'\s+\]', ']', s)
    s = re.sub(r'\[\s+', '[', s)
    return s.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Base Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """Base class; subclasses implement extraction and pairwise equivalence."""

    def _is_number(self, s) -> Tuple[bool, str]:
        try:
            res = float(s)
            return True, str(res)
        except Exception:
            pass
        try:
            import unicodedata
            res = unicodedata.numeric(s)
            return True, str(res)
        except Exception:
            pass
        return False, None

    def _last_boxed_only_string(self, string):
        if string is None:
            return None
        assert isinstance(string, str)
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i, right_brace_idx, num_open = idx, None, 0
        while i < len(string):
            if string[i] == "{":
                num_open += 1
            if string[i] == "}":
                num_open -= 1
                if num_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        if right_brace_idx is None:
            return None
        return string[idx: right_brace_idx + 1]

    def _remove_boxed(self, s):
        if s is None:
            return None
        assert isinstance(s, str)
        results = re.findall(r'\\boxed\{(.*)\}', s, re.DOTALL)
        return results[-1] if results else None

    def validate_model_completion(self, completion: str) -> bool:
        return True

    def validate_model_answer(self, answer: str) -> bool:
        return answer is not None

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        if not self.validate_model_answer(answer_a) or not self.validate_model_answer(answer_b):
            return None
        return (
            self._check_answers_equiv(answer_a, answer_b)
            or self._check_answers_equiv(answer_b, answer_a)
        )

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        raise NotImplementedError

    def extract_answer_from_model_completion(self, completion: str) -> str:
        if not self.validate_model_completion(completion):
            return None
        return self._extract_answer_from_model_completion(completion)

    def _extract_answer_from_model_completion(self, completion: str) -> str:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-specific evaluators
# ──────────────────────────────────────────────────────────────────────────────

class GSM8KEvaluator(Evaluator):
    """Numeric equality of ``\\boxed{}`` answers."""

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return None
        ok_a, num_a = self._is_number(answer_a)
        ok_b, num_b = self._is_number(answer_b)
        return (num_a == num_b) if (ok_a and ok_b) else False

    def validate_model_completion(self, completion: str) -> bool:
        return isinstance(completion, str) and "\\boxed" in completion

    def _extract_answer_from_model_completion(self, completion: str):
        ans = self._remove_boxed(completion)
        return ans if ans else None


def _numeric_equiv(a: str, b: str, tol: float = 1e-2) -> bool:
    """Try to evaluate both expressions numerically and compare."""
    def _to_float(s):
        try:
            return float(s)
        except Exception:
            pass
        try:
            from sympy import N as sympyN
            from sympy.parsing.latex import parse_latex
            return float(sympyN(parse_latex(s.replace('\\\\', '\\'))))
        except Exception:
            pass
        try:
            from latex2sympy2 import latex2sympy
            from sympy import N as sympyN
            return float(sympyN(latex2sympy(s)))
        except Exception:
            pass
        return None

    fa, fb = _to_float(a), _to_float(b)
    if fa is not None and fb is not None:
        if fb != 0:
            return abs(fa - fb) / abs(fb) < tol
        return abs(fa - fb) < tol
    return False


class MATHEvaluator(Evaluator):
    """LaTeX-aware equivalence with a numeric fallback (rel. tol. 1e-2)."""

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return None
        if answer_a == "" or answer_b == "":
            return False

        a = answer_a.strip()
        b = answer_b.strip()

        # 1. Exact match
        if a.lower() == b.lower():
            return True

        # 2. Normalised LaTeX match (whitespace / \\left\\right tolerant)
        na, nb = _normalise_latex(a), _normalise_latex(b)
        if na.lower() == nb.lower():
            return True

        # 3. Symbolic / numeric equivalence via latex_equiv
        try:
            if latex_equiv(a, b):
                return True
        except Exception:
            pass

        # 4. latex_equiv on normalised forms
        try:
            if latex_equiv(na, nb):
                return True
        except Exception:
            pass

        # 5. Numeric equivalence (handles \frac{1}{12} == 0.0833...)
        if _numeric_equiv(a, b):
            return True

        return False

    def validate_model_completion(self, completion: str) -> bool:
        return isinstance(completion, str) and "\\boxed" in completion

    def _extract_answer_from_model_completion(self, completion: str):
        box = self._last_boxed_only_string(completion)
        ans = self._remove_boxed(box)
        return ans if ans else None


class OlympiadEvaluator(Evaluator):
    """LaTeX-aware equivalence without the numeric fallback."""

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return None
        if answer_a == "" or answer_b == "":
            return False
        a, b = answer_a.strip(), answer_b.strip()
        if a.lower() == b.lower():
            return True
        na, nb = _normalise_latex(a), _normalise_latex(b)
        if na.lower() == nb.lower():
            return True
        try:
            if latex_equiv(a, b):
                return True
        except Exception:
            pass
        try:
            return latex_equiv(na, nb)
        except Exception:
            return False

    def _extract_answer_from_model_completion(self, completion: str):
        box = self._last_boxed_only_string(completion)
        ans = self._remove_boxed(box)
        return ans if ans else None


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

EvaluatorMapping = {
    "GSM8K":         GSM8KEvaluator,
    "MATH":          MATHEvaluator,
    "MATH500":       MATHEvaluator,
    "AMC":           MATHEvaluator,
    "AMC2023":       MATHEvaluator,
    "AIME":          MATHEvaluator,
    "AIME2024":      MATHEvaluator,
    "OlympiadBench": OlympiadEvaluator,
}


def get_evaluator(task_name: str = "MATH500") -> Evaluator:
    """Instantiate the evaluator registered for ``task_name``."""
    if task_name in EvaluatorMapping:
        return EvaluatorMapping[task_name]()
    if "MATH" in task_name:
        return MATHEvaluator()
    raise ValueError(f"Task '{task_name}' not in EvaluatorMapping")
