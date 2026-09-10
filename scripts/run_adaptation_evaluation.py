"""Interactive three-policy comparison; serial processes keep GPU memory bounded."""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.adaptation_eval_records import read_training, export_comparison
from smolvla_grounding.benchmark_records import ask_indices, validate_run_name
from smolvla_grounding.protocol import atomic_json, file_sha256, fingerprint, run_lock


def main():
    settings = json.loads((ROOT / "configs/adaptation_evaluation.json").read_text())
    baseline = json.loads((ROOT / settings["baseline_run"] / "identity.json").read_text())
    rows = [r for r in baseline["instructions"] if r["id"] in settings["instruction_ids"]]
    if [r["id"] for r in rows] != settings["instruction_ids"] or any(r["goal_id"] != "original_bowl" for r in rows):
        raise RuntimeError("Expected A, its repeat control and the two development paraphrases")
    print("[1/3] Check paired training evidence and freeze selected final checkpoints", flush=True)
    training = read_training(ROOT, settings)
    names = ["configs/adaptation_evaluation.json", "scripts/run_adaptation_evaluation.py",
             "scripts/evaluate_adaptation_model.py", "scripts/run_adaptation_evaluation.sh",
             *["src/smolvla_grounding/" + n + ".py" for n in (
                 "adaptation_eval_records", "adaptation_eval_runtime", "adaptation_records", "training_policy",
                 "benchmark_records", "benchmark_runtime", "benchmark_episode", "protocol", "paired_env",
                 "libero_pairing", "rollout_records", "diagnostic_records", "diagnostic_policy")]]
    provenance = {"settings": settings, "training": training, "baseline_fingerprint": fingerprint(baseline),
                  "sources": {n: file_sha256(ROOT / n) for n in names}}
    for row in rows:
        print("  " + row["id"] + ": " + row["text"], flush=True)
    while True:
        try:
            name = validate_run_name(input("Evaluation run name [task0_adaptation_eval]: ").strip() or "task0_adaptation_eval")
            break
        except ValueError as exc:
            print(str(exc), flush=True)
    directory = ROOT / settings["output_root"] / "runs" / name
    with run_lock(directory):
        path = directory / "experiment.json"
        if path.exists():
            experiment = json.loads(path.read_text())
            if experiment["provenance"] != provenance:
                raise RuntimeError("Checkpoint/code/config changed; preserve this run and choose a new name")
            print(json.dumps({k: experiment[k] for k in ("initial_states", "policy_seeds")}), flush=True)
            if input("Continue and verify completed episodes? [y/N]: ").strip().lower() != "y":
                return
        else:
            experiment = {"provenance": provenance, "initial_states": ask_indices("Initial-state indices", "0,1,2"),
                          "policy_seeds": ask_indices("Policy seeds", "0"), "instructions": rows,
                          "environment_seed": baseline["config"]["environment_seed"],
                          "suite": baseline["config"]["suite"], "task_id": baseline["config"]["task_id"],
                          "horizon": baseline["config"]["environment"]["episode_length"]}
            atomic_json(path, experiment)
        count = len(experiment["initial_states"]) * len(experiment["policy_seeds"]) * len(rows) * len(settings["models"])
        print(f"[2/3] {count} episodes: three models, four instructions per state/seed; one GPU policy at a time.", flush=True)
        print("Default 36 episodes may take about 2-3 hours based on prior rollouts; precision, imports and I/O can change this. Each completed episode is resumable.", flush=True)
        for model in settings["models"]:
            print("Start model: " + model, flush=True)
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/evaluate_adaptation_model.py")],
                env=dict(os.environ, SMOLVLA_ADAPTATION_EVAL_DIR=str(directory), SMOLVLA_ADAPTATION_EVAL_MODEL=model), cwd=ROOT, check=True)
        print("[3/3] Export paired before/after and augmentation-versus-original-tuning outcomes", flush=True)
        results = {m: json.loads((directory / m / "episodes.json").read_text()) for m in settings["models"]}
        final = export_comparison(directory, results, experiment, rows)
        if not final["all_complete"]:
            raise RuntimeError("Some model/state/language groups remain incomplete")
        print(f"PASS: {count} episodes across three precision-matched models completed.", flush=True)
        print(json.dumps(final["paired_differences"], indent=2), flush=True)
        print("Saved " + (directory / "comparison.json").relative_to(ROOT).as_posix(), flush=True)
        print("Development results only. Review videos; this is not a full LIBERO benchmark or an untouched test.", flush=True)


if __name__ == "__main__":
    main()
