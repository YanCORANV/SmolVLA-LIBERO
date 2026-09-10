"""Compare final 1000-update policies with the 200-update development reference."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.configured_evaluation import run_configured_evaluation


if __name__ == "__main__":
    run_configured_evaluation("configs/training_budget_evaluation.json", [
        "scripts/run_training_budget_evaluation.py", "scripts/run_training_budget_evaluation.sh"])
