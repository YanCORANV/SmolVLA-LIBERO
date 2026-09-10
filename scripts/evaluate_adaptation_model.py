"""One GPU policy process, using the established reset/rollout/video implementation."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.benchmark_records import BenchmarkJournal, export_benchmark, trial_id
from smolvla_grounding.protocol import atomic_json, file_sha256, fingerprint, run_lock


def main():
    directory = Path(os.environ["SMOLVLA_ADAPTATION_EVAL_DIR"])
    model = os.environ["SMOLVLA_ADAPTATION_EVAL_MODEL"]
    experiment = json.loads((directory / "experiment.json").read_text())
    provenance = experiment["provenance"]
    settings = provenance["settings"]
    if model not in settings["models"]:
        raise ValueError("Unexpected model")
    out = directory / model
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    execution = {"passed": False, "model": model, "stages": {}}

    def stage(name):
        execution["stages"][name] = time.perf_counter() - started
        print(f"  [{model}] elapsed {execution['stages'][name]:.1f} s: {name}", flush=True)

    with run_lock(out):
        try:
            for name, expected in provenance["sources"].items():
                if file_sha256(ROOT / name) != expected:
                    raise RuntimeError("Evaluation source changed: " + name)
            baseline = json.loads((ROOT / settings["baseline_run"] / "identity.json").read_text())
            if fingerprint(baseline) != provenance["baseline_fingerprint"]:
                raise RuntimeError("Baseline provenance changed")
            print(f"[{model} 1/4] Verify official runtime and selected checkpoint; cold imports may take minutes", flush=True)
            from smolvla_grounding.adaptation_eval_runtime import AdaptationEvalRuntime
            runtime = AdaptationEvalRuntime(ROOT, baseline, model, provenance["training"]["models"][model], experiment["instructions"])
            if max(experiment["initial_states"]) >= runtime.state_count:
                raise ValueError(f"State indices must be below {runtime.state_count}; select a new run name")
            identity = {"experiment": experiment, "runtime": runtime.provenance, "model": model}
            journal = BenchmarkJournal(out, identity)
            shared = BenchmarkJournal(directory / "references", {"experiment": experiment, "runtime": runtime.provenance})
            stage("runtime_verified")
            from smolvla_grounding.benchmark_episode import verify_reference, run_episode
            from smolvla_grounding.libero_pairing import require_matching_state
            print(f"[{model} 2/4] Verify shared physical reset references", flush=True)
            references = {}
            for state in experiment["initial_states"]:
                key = f"state_{state:04d}"
                reference = shared.completed(key)
                if reference is None:
                    attempt = shared.new_attempt(key)
                    config = deepcopy(runtime.config)
                    config["init_state_index"] = state
                    reference, artifacts = verify_reference(runtime, config, attempt)
                    shared.finish(key, reference, artifacts)
                references[state] = reference
            stage("references_verified")
            saved = {trial_id(s, p, r["id"]): journal.completed(trial_id(s, p, r["id"]))
                     for s in experiment["initial_states"] for p in experiment["policy_seeds"] for r in runtime.rows}
            print(f"[{model} 3/4] Load policy with exact parameter/normalization checks", flush=True)
            if any(r is None for r in saved.values()):
                atomic_json(out / "model_load.json", runtime.load_policy())
            stage("policy_ready")
            verified = {k: v for k, v in saved.items() if v is not None}
            export_benchmark(out, list(verified.values()), experiment, runtime.rows)
            print(f"[{model} 4/4] Run fixed-horizon episodes; save each completed case", flush=True)
            position = 0
            for state in experiment["initial_states"]:
                for seed in experiment["policy_seeds"]:
                    config = deepcopy(runtime.config)
                    config.update(init_state_index=state, policy_seed=seed)
                    rng_key = f"rng_state_{state:04d}_seed_{seed:04d}"
                    reference_rng = shared.completed(rng_key)
                    for row in runtime.rows:
                        position += 1
                        key = trial_id(state, seed, row["id"])
                        result = saved[key]
                        print(f"  [{position}/{len(saved)}] {model}: state={state}, seed={seed}, {row['id']}", flush=True)
                        if result is None:
                            attempt = journal.new_attempt(key)
                            try:
                                result, artifacts = run_episode(runtime, config, row, references[state], reference_rng, attempt, out)
                                journal.finish(key, result, artifacts)
                            except BaseException as exc:
                                atomic_json(attempt / "error.json", {"type": type(exc).__name__, "message": str(exc)})
                                raise
                        if (result["initial_state_index"], result["policy_seed"], result["instruction_id"]) != (state, seed, row["id"]):
                            raise RuntimeError("Completed episode identity mismatch")
                        require_matching_state(references[state]["initial_hashes"], result["initial_hashes"])
                        if reference_rng is None:
                            reference_rng = result["initial_rng_hashes"]
                            attempt = shared.new_attempt(rng_key)
                            atomic_json(attempt / "rng_hashes.json", reference_rng)
                            shared.finish(rng_key, reference_rng, [attempt / "rng_hashes.json"])
                        require_matching_state(reference_rng, result["initial_rng_hashes"])
                        verified[key] = result
                        summary = export_benchmark(out, list(verified.values()), experiment, runtime.rows)
                        print("  Saved/reverified original-goal outcome: " + json.dumps(result["goals"]["original_bowl"]), flush=True)
                        stage(key)
            if not summary["all_complete"]:
                raise RuntimeError("Model evaluation incomplete")
            execution["passed"] = True
            print(f"PASS: {model}, {len(saved)} verified episodes; repeat outcome disagreements: {summary['repeat_ever_disagreement_groups']}.", flush=True)
        except BaseException as exc:
            execution["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            execution["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(out / f"execution_{stamp}.json", execution)


if __name__ == "__main__":
    main()
