"""Analyze only after all 75 development-pilot rollouts are committed."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.multitask_pilot_analysis import analyze

if __name__ == "__main__":
    analyze()
