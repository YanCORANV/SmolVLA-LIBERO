"""Execute frozen confirmatory held-out language evaluation on Task 0."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.held_out_evaluator import run_held_out_evaluation
from smolvla_grounding.held_out_summary import summarize_held_out_run


if __name__ == "__main__":
    matrix_path = "configs/held_out_evaluation/confirmatory_matrix.json"
    run_held_out_evaluation(matrix_path, default_run_name="task0_confirmatory_heldout_eval")

    import json
    cfg = json.loads((ROOT / matrix_path).read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["output_root"] / "runs" / "task0_confirmatory_heldout_eval"
    if out_dir.exists():
        summarize_held_out_run(out_dir)
