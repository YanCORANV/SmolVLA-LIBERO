"""Run the registered 30-rollout canonical competence screen."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.multitask_evaluator import run_canonical_screen

if __name__ == "__main__":
    run_canonical_screen()
