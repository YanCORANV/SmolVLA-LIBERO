"""Interactive multi-initial-state task-0 benchmark using the official runtime."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.benchmark_records import (
    BenchmarkJournal, ask_indices, export_benchmark, trial_id, validate_experiment, validate_run_name,
)
from smolvla_grounding.protocol import atomic_json, file_sha256, fingerprint, run_lock


def main():
    if platform.system() != "Linux":
        raise RuntimeError("Run in the validated WSL/Linux environment")
    if os.environ.get("MUJOCO_GL", "egl") != "egl":
        raise RuntimeError("Use the validated EGL renderer")
    os.environ["MUJOCO_GL"] = "egl"
    settings = json.loads((ROOT / "configs/state_benchmark.json").read_text())
    source = ROOT / settings["source_run"]
    baseline = json.loads((source / "identity.json").read_text())
    old_summary = json.loads((source / "summary.json").read_text())
    if not old_summary["all_trials_complete"]:
        raise RuntimeError("Complete the pilot before expanding states")
    while True:
        try:
            name = validate_run_name(input("Run name [task0_states]: ").strip() or "task0_states")
            break
        except ValueError as exc:
            print(str(exc), flush=True)
    directory = ROOT / settings["output_root"] / "runs" / name
    with run_lock(directory):
        experiment_path = directory / "experiment.json"
        if experiment_path.exists():
            experiment = json.loads(experiment_path.read_text())
            print("Saved experiment: " + json.dumps(experiment, indent=2), flush=True)
            if input("Continue this experiment and verify completed work? [y/N]: ").strip().lower() != "y":
                print("Existing experiment preserved.", flush=True)
                return
            if experiment["source_run_fingerprint"] != fingerprint(baseline):
                raise RuntimeError("Source pilot changed; choose a new run name")
        else:
            print("Choose exact official initial-state indices and policy seeds; each combination has six trials.", flush=True)
            states = ask_indices("Initial-state indices", "0,1,2")
            seeds = ask_indices("Policy seeds", "0")
            experiment = {"run_name": name, "source_run_fingerprint": fingerprint(baseline),
                          "initial_states": states, "policy_seeds": seeds,
                          "environment_seed": baseline["config"]["environment_seed"],
                          "instruction_order": [r["id"] for r in baseline["instructions"]],
                          "horizon": baseline["config"]["environment"]["episode_length"],
                          "suite": baseline["config"]["suite"], "task_id": baseline["config"]["task_id"]}
            atomic_json(experiment_path, experiment)
        validate_experiment(experiment, baseline)
        n_trials = len(experiment["initial_states"]) * len(experiment["policy_seeds"]) * len(baseline["instructions"])
        nominal = n_trials * statistics.median(r["rollout_seconds"] for r in old_summary["episodes"]) / 60
        print(f"Selected {n_trials} episodes, {experiment['horizon']} policy steps each, one GPU/environment at a time.", flush=True)
        print(f"Prior rollout throughput suggests about {nominal:.0f} minutes of rollout alone; imports, reset checks and I/O are additional.", flush=True)
        print("Whole-command duration is uncertain; each stage is timed. Output: " + directory.relative_to(ROOT).as_posix(), flush=True)
        started = time.perf_counter()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        execution = {"passed": False, "stage_elapsed_seconds": {},
                     "execution_logs": os.environ.get("SMOLVLA_BENCHMARK_LOG_DIR"), "expected_episodes": n_trials}

        def stage(label):
            elapsed = time.perf_counter() - started
            execution["stage_elapsed_seconds"][label] = elapsed
            print(f"  Elapsed {elapsed:.1f} s: {label}", flush=True)

        try:
            print("[1/5] Verify audited runtime, packages, assets and checkpoint (cold imports may be slow)", flush=True)
            from smolvla_grounding.benchmark_runtime import BenchmarkRuntime
            runtime = BenchmarkRuntime(ROOT, baseline)
            if max(experiment["initial_states"]) >= runtime.state_count:
                raise ValueError(f"Initial-state indices must be below {runtime.state_count}; preserve this selection and choose a new run name")
            if experiment["instruction_order"] != [r["id"] for r in runtime.rows]:
                raise RuntimeError("Saved instruction order differs from the source pilot")
            source_names = ("scripts/run_state_benchmark.py", "scripts/run_state_benchmark.sh", "configs/state_benchmark.json",
                            "src/smolvla_grounding/benchmark_records.py", "src/smolvla_grounding/benchmark_runtime.py",
                            "src/smolvla_grounding/benchmark_episode.py", "src/smolvla_grounding/diagnostic_records.py",
                            "src/smolvla_grounding/diagnostic_policy.py")
            identity = {"experiment": experiment, "settings": settings, "runtime": runtime.provenance,
                        "sources": {name: file_sha256(ROOT / name) for name in source_names}}
            journal = BenchmarkJournal(directory, identity)
            stage("runtime_verified")
            from smolvla_grounding.benchmark_episode import verify_reference, run_episode
            from smolvla_grounding.libero_pairing import require_matching_state
            from smolvla_grounding.diagnostic_records import compare_traces
            import numpy as np

            print("[2/5] Verify two fresh resets per selected initial state before policy episodes", flush=True)
            references = {}
            for state in experiment["initial_states"]:
                key = f"reference_state_{state:04d}"
                reference = journal.completed(key)
                if reference is None:
                    attempt = journal.new_attempt(key)
                    config = deepcopy(runtime.config)
                    config["init_state_index"] = state
                    try:
                        reference, artifacts = verify_reference(runtime, config, attempt)
                        journal.finish(key, reference, artifacts)
                    except BaseException as exc:
                        atomic_json(attempt / "error.json", {"type": type(exc).__name__, "message": str(exc)})
                        raise
                else:
                    print(f"  State {state}: verified saved reset evidence", flush=True)
                references[state] = reference
            stage("all_initial_state_checks_passed")

            saved = {trial_id(state, seed, row["id"]): journal.completed(trial_id(state, seed, row["id"]))
                     for state in experiment["initial_states"] for seed in experiment["policy_seeds"] for row in runtime.rows}
            print("[3/5] Load one official policy for all pending episodes", flush=True)
            if any(r is None for r in saved.values()):
                atomic_json(directory / "model_load.json", runtime.load_policy())
            else:
                print("  All episodes complete; skip model loading", flush=True)
            stage("policy_ready")

            print("[4/5] Run matched state/seed groups and save each verified episode", flush=True)
            results, controls = [], []
            # Preserve all already verified rows in progress exports, even if a
            # subsequent interrupted run stops before reaching their group in order.
            verified_results = {key: value for key, value in saved.items() if value is not None}
            export_benchmark(directory, list(verified_results.values()), experiment, runtime.rows)
            for state in experiment["initial_states"]:
                for seed in experiment["policy_seeds"]:
                    config = deepcopy(runtime.config)
                    config.update(init_state_index=state, policy_seed=seed)
                    group = []
                    reference_rng = None
                    for row in runtime.rows:
                        key = trial_id(state, seed, row["id"])
                        result = saved[key]
                        print(f"  [{len(results) + 1}/{n_trials}] state={state}, policy_seed={seed}, {row['id']}", flush=True)
                        if result is None:
                            attempt = journal.new_attempt(key)
                            try:
                                result, artifacts = run_episode(runtime, config, row, references[state], reference_rng, attempt, directory)
                                journal.finish(key, result, artifacts)
                            except BaseException as exc:
                                atomic_json(attempt / "error.json", {"type": type(exc).__name__, "message": str(exc)})
                                raise
                            print("  Saved requested goal: " + json.dumps(result["goals"][result["goal_id"]]), flush=True)
                        else:
                            print("  Reusing verified completed episode", flush=True)
                        if result["initial_state_index"] != state or result["policy_seed"] != seed or result["instruction_id"] != row["id"]:
                            raise RuntimeError("Completed episode identity mismatch")
                        require_matching_state(references[state]["initial_hashes"], result["initial_hashes"])
                        if reference_rng is None:
                            reference_rng = result["initial_rng_hashes"]
                        require_matching_state(reference_rng, result["initial_rng_hashes"])
                        group.append(result)
                        results.append(result)
                        verified_results[key] = result
                        export_benchmark(directory, list(verified_results.values()), experiment, runtime.rows)
                    original = next(r for r in group if r["condition"] == "original")
                    repeat = next(r for r in group if r["condition"] == "original_repeat")
                    traces = [[json.loads(line) for line in (directory / r["inference_log"]).read_text().splitlines()]
                              for r in (original, repeat)]
                    with np.load(directory / original["trajectory"], allow_pickle=False) as a, \
                         np.load(directory / repeat["trajectory"], allow_pickle=False) as b:
                        control = {"initial_state_index": state, "policy_seed": seed,
                                   "actions_equal": bool(np.array_equal(a["action"], b["action"])),
                                   "sim_states_equal": bool(np.array_equal(a["sim_state"], b["sim_state"])),
                                   "max_absolute_action_difference": float(np.abs(a["action"] - b["action"]).max()),
                                   "trace_comparison": compare_traces(*traces)}
                    controls.append(control)
                    atomic_json(directory / "repeat_controls.json", controls)
                    stage(f"state_{state}_seed_{seed}_complete")
            print("[5/5] Export complete-group outcomes and paired paraphrase differences", flush=True)
            final = export_benchmark(directory, results, experiment, runtime.rows)
            if not final["all_complete"]:
                raise RuntimeError("Expected every selected state/seed group to be complete")
            execution["passed"] = True
            print(f"PASS: {n_trials} episodes in {final['completed_groups']} matched groups completed.", flush=True)
            print(json.dumps({"per_instruction": final["per_instruction"],
                              "mean_paired_paraphrase_minus_original": final["mean_paired_paraphrase_minus_original"],
                              "repeat_ever_disagreement_groups": final["repeat_ever_disagreement_groups"]}, indent=2), flush=True)
            print("Saved " + (directory / "summary.json").relative_to(ROOT).as_posix(), flush=True)
            print("Changed-goal scores remain provisional pending positive-case review. No fine-tuning was performed.", flush=True)
        except BaseException as exc:
            execution["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            execution["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(directory / f"execution_{stamp}.json", execution)


if __name__ == "__main__":
    main()
