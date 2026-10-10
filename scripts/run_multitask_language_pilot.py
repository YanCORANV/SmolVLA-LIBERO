"""Run the frozen 75-rollout development pilot with committed-cell resume."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.multitask_pilot_evaluator import run_pilot

if __name__ == "__main__":
    run_pilot()
