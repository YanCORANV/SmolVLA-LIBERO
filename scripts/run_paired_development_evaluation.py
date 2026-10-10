"""Run the frozen 60-rollout paired-flow development evaluation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.paired_development_evaluator import run_paired_development_evaluation


if __name__ == "__main__":
    run_paired_development_evaluation()