"""Execute model-selection development evaluation on States 23-32."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.development_evaluator import run_development_evaluation


if __name__ == "__main__":
    matrix_path = "configs/development_evaluation/dev_matrix.json"
    run_development_evaluation(matrix_path, default_run_name="task0_anchored_dev_eval")
