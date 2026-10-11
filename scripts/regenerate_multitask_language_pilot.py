"""Verify the committed pilot tables and regenerate figures without raw rollouts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.multitask_pilot_analysis import regenerate_published

if __name__ == "__main__":
    regenerate_published()
