"""Breadth-depth test-time refinement for LLM math reasoning.

Method ("CIMR" in code):
    cimr_prompts     generator / critic / corrector role prompts
    cimr_step        one refinement depth (continuation, critique, correction)
    cimr_rollout     N rollouts x D depths for one problem, with resume
    cimr_aggregator  answer extraction and plurality vote
    cimr_inference   orchestration and the per-run results store

Shared infrastructure:
    model_wrapper      Hugging Face generation wrapper
    dataset_loader     benchmarks and gold answers
    evaluator_classes  answer extraction / equivalence (uses math_toolkit)
    inference_methods  baselines: greedy, Maj@N, Best-of-N, log-prob selection
    evaluator          resumable evaluation loop for the baselines
    verifier           process reward model (Best-of-N baseline, diagnostics)
    clustering         sampling-redundancy clustering (motivation analysis)
"""
