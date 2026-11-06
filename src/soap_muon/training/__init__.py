"""Training loop, evaluation, checkpointing, and metrics."""

from soap_muon.training.evaluation import EvaluationResult, evaluate_next_token_prediction
from soap_muon.training.benchmark import TrainingBenchmarkResult, benchmark_training
from soap_muon.training.optimizers import OptimizerBundle, build_optimizer_bundle

__all__ = [
    "EvaluationResult",
    "OptimizerBundle",
    "TrainingBenchmarkResult",
    "benchmark_training",
    "build_optimizer_bundle",
    "evaluate_next_token_prediction",
]
