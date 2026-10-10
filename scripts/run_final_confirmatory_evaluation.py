"""Execute the frozen 180-rollout Task-0 final confirmatory evaluation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.final_confirmatory_evaluator import run_final_confirmatory_evaluation


if __name__ == "__main__":
    run_final_confirmatory_evaluation()