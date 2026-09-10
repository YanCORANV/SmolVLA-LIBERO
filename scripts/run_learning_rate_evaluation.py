"""Evaluate two new policies against verified cached untuned development episodes."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.adaptation_eval_records import read_training, export_comparison, compare_models
from smolvla_grounding.benchmark_records import BenchmarkJournal, export_benchmark, trial_id, validate_run_name
from smolvla_grounding.learning_rate_eval_records import import_completed, require_same_initial_parameters
from smolvla_grounding.protocol import atomic_json, file_sha256, fingerprint, run_lock


def main():
    settings_path = "configs/learning_rate_evaluation.json"
    settings = json.loads((ROOT / settings_path).read_text())
    reference = ROOT / settings["reference_evaluation"]
    previous = json.loads((reference / "experiment.json").read_text())
    old_report = json.loads((reference / "comparison.json").read_text())
    if not old_report["all_complete"]:
        raise RuntimeError("Complete the reference evaluation first")
    for name, expected in previous["provenance"]["sources"].items():
        if file_sha256(ROOT / name) != expected:
            raise RuntimeError("Reference evaluator source changed: " + name)
    baseline = json.loads((ROOT / settings["baseline_run"] / "identity.json").read_text())
    if fingerprint(baseline) != previous["provenance"]["baseline_fingerprint"]:
        raise RuntimeError("Official baseline identity changed")
    print("[1/4] Verify lower-learning-rate training evidence and unchanged initial weights", flush=True)
    training = read_training(ROOT, settings)
    if read_training(ROOT, previous["provenance"]["settings"]) != previous["provenance"]["training"]:
        raise RuntimeError("Reference training/checkpoint evidence changed")
    require_same_initial_parameters(ROOT, previous, training)
    rows = previous["instructions"]
    if [r["id"] for r in rows] != settings["instruction_ids"]:
        raise RuntimeError("Instruction catalog differs from the reference")
    source_identity = json.loads((reference / "untuned_fp32/identity.json").read_text())
    if source_identity["experiment"] != previous or source_identity["model"] != "untuned_fp32":
        raise RuntimeError("Reference episode journal has a different experiment/model")
    reference_load = json.loads((reference / "untuned_fp32/model_load.json").read_text())
    if not all(reference_load[k] for k in ("strict_load", "all_parameter_bytes_equal", "normalization_matches_original")):
        raise RuntimeError("Reference baseline reload checks did not pass")
    source_refs_identity = json.loads((reference / "references/identity.json").read_text())
    if source_refs_identity != {"experiment": previous, "runtime": source_identity["runtime"]}:
        raise RuntimeError("Reference reset journal has different runtime provenance")
    names = list(previous["provenance"]["sources"])
    names += [settings_path, "scripts/run_learning_rate_evaluation.py", "scripts/run_learning_rate_evaluation.sh",
              "src/smolvla_grounding/learning_rate_eval_records.py"]
    provenance = {"settings": settings, "training": training, "baseline_fingerprint": fingerprint(baseline),
                  "sources": {n: file_sha256(ROOT / n) for n in names},
                  "reference": {n: file_sha256(reference / n) for n in (
                      "experiment.json", "comparison.json", "untuned_fp32/identity.json", "untuned_fp32/model_load.json", "references/identity.json")}}
    while True:
        try:
            name = validate_run_name(input(f"Evaluation run name [{settings['default_run_name']}]: ").strip() or settings["default_run_name"])
            break
        except ValueError as exc:
            print(str(exc), flush=True)
    directory = ROOT / settings["output_root"] / "runs" / name
    if directory == reference:
        raise ValueError("Keep the reference evaluation; choose a new name")
    with run_lock(directory):
        experiment = deepcopy(previous)
        experiment["provenance"] = provenance
        path = directory / "experiment.json"
        if path.exists():
            if json.loads(path.read_text()) != experiment:
                raise RuntimeError("Saved experiment differs; preserve it and choose a new name")
            if input("Continue and verify completed episodes? [y/N]: ").strip().lower() != "y":
                return
        else:
            atomic_json(path, experiment)
        source = BenchmarkJournal(reference / "untuned_fp32", source_identity)
        destination = BenchmarkJournal(directory / "untuned_fp32", {**source_identity, "experiment": experiment})
        source_refs = BenchmarkJournal(reference / "references", source_refs_identity)
        destination_refs = BenchmarkJournal(directory / "references", {**source_refs_identity, "experiment": experiment})
        print("[2/4] Import verified untuned episodes and shared reset/RNG references", flush=True)
        untuned = []
        for state in experiment["initial_states"]:
            import_completed(source_refs, destination_refs, f"state_{state:04d}")
            for seed in experiment["policy_seeds"]:
                import_completed(source_refs, destination_refs, f"rng_state_{state:04d}_seed_{seed:04d}")
                for row in rows:
                    key = trial_id(state, seed, row["id"])
                    print("  Verify/import " + key, flush=True)
                    untuned.append(import_completed(source, destination, key))
        export_benchmark(directory / "untuned_fp32", untuned, experiment, rows)
        atomic_json(directory / "untuned_fp32/model_load.json", reference_load)
        atomic_json(directory / "baseline_reuse.json", {"source": settings["reference_evaluation"],
            "episodes_reused": len(untuned), "new_baseline_rollouts": 0, "source_identity_sha256": file_sha256(reference / "untuned_fp32/identity.json"),
            "same_initial_parameter_bytes": True, "scope": "Reused reference observations; not independent repeated baseline trials"})
        print(f"[3/4] Run {2 * len(untuned)} new episodes with the unchanged worker; {len(untuned)} baseline episodes reused.", flush=True)
        print("Expect roughly 1.5-2 hours from earlier throughput; startup and I/O may vary. Same states/seeds/wordings as the first round.", flush=True)
        for model in settings["models"][1:]:
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/evaluate_adaptation_model.py")], cwd=ROOT,
                env=dict(os.environ, SMOLVLA_ADAPTATION_EVAL_DIR=str(directory), SMOLVLA_ADAPTATION_EVAL_MODEL=model), check=True)
        print("[4/4] Export matched outcomes and differences from the first training round", flush=True)
        results = {m: json.loads((directory / m / "episodes.json").read_text()) for m in settings["models"]}
        final = export_comparison(directory, results, experiment, rows)
        if not final["all_complete"]:
            raise RuntimeError("Evaluation incomplete")
        # Recompute previous round's small records for descriptive cross-round deltas.
        previous_results = {m: json.loads((reference / m / "episodes.json").read_text()) for m in settings["models"]}
        if compare_models(previous_results, previous, rows) != old_report:
            raise RuntimeError("Previous comparison no longer matches its episode records")
        changes = []
        for model in settings["models"][1:]:
            for old, new in zip(sorted(previous_results[model], key=lambda r: (r['initial_state_index'], r['policy_seed'], r['instruction_id'])),
                                sorted(results[model], key=lambda r: (r['initial_state_index'], r['policy_seed'], r['instruction_id'])), strict=True):
                keys = ("initial_state_index", "policy_seed", "instruction_id", "initial_hashes", "initial_rng_hashes")
                if any(old[k] != new[k] for k in keys):
                    raise RuntimeError("Cross-round cases are not paired")
                changes.append({"model": model, **{k: new[k] for k in keys[:3]},
                    "previous_ever": old["goals"]["original_bowl"]["ever"], "new_ever": new["goals"]["original_bowl"]["ever"],
                    "previous_final": old["goals"]["original_bowl"]["final"], "new_final": new["goals"]["original_bowl"]["final"]})
        atomic_json(directory / "learning_rate_changes.json", {"reference": settings["reference_evaluation"], "cases": changes})
        print("PASS: lower-learning-rate evaluation complete; 24 new episodes and 12 verified reference episodes at the default case set.", flush=True)
        print(json.dumps(final["paired_differences"], indent=2), flush=True)
        print("Saved " + (directory / "comparison.json").relative_to(ROOT).as_posix(), flush=True)
        print("Development comparison only; reused baseline episodes are not new independent evidence.", flush=True)


if __name__ == "__main__":
    main()
