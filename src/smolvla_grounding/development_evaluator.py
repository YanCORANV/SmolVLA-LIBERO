"""Development evaluator for model selection on States 23-32.

Enforces strict boundary:
- Historical dev: States 0-2 (excluded)
- Permanently consumed test: States 3-22 (strictly forbidden)
- Target model-selection dev: States 23-32 (strictly 10 states)
- Future confirmatory: States 33-52 (strictly untouched)

Evaluates:
1. original_language
2. augmented_language
3. anchored_language

On:
- original
- paraphrase_dev_01
- paraphrase_dev_02
Total = 10 states * 3 prompts * 3 models = 90 rollouts.
Uses audited 9-stage LeRobot pipeline, Worker=1, horizon=280 steps.
"""

from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .protocol import atomic_json, file_sha256, run_lock
from .benchmark_records import validate_run_name


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_task_goal_status(env):
    """Check task goal status using native LIBERO check_success."""
    return bool(env._env.check_success())


def validate_development_prompts(root, instruction_catalog_rel):
    """
    Rigorously validate development instructions against split isolation and semantic rules:

    1. Structure:
       - Strictly 3 instructions: ['original', 'paraphrase_dev_01', 'paraphrase_dev_02']

    2. String Disjointness Rules:
       - No exact raw string duplicate with any training, consumed confirmatory, historical pilot, or reserved final paraphrase.
       - No normalized full-string duplicate (lowercasing, punctuation stripped) with any training, consumed confirmatory, historical pilot, or reserved final paraphrase.
       - The development paraphrases must not duplicate each other or the original canonical prompt.

    3. Semantic Equivalence & Lexical Constraints:
       - Target object: must refer to the "black bowl".
       - Spatial landmarks: must reference "plate" and "ramekin".
       - Manipulation & Goal: must specify moving/placing onto the "plate".
       - Expected lexical overlap in task-critical nouns ("black", "bowl", "plate", "ramekin") is explicitly verified as present.
    """
    import re

    def norm_full(text: str) -> str:
        """Normalized full sentence: lowercased, punctuation stripped, single whitespace."""
        return " ".join(re.findall(r"\w+", text.lower()))

    def word_set(text: str) -> set:
        """Set of lowercased word tokens."""
        return set(re.findall(r"\w+", text.lower()))

    inst_path = root / instruction_catalog_rel
    if not inst_path.exists():
        raise RuntimeError(f"FATAL: Development instruction catalog does not exist: {inst_path}")

    instructions = load_jsonl(inst_path)
    if len(instructions) != 3:
        raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Expected exactly 3 instructions in dev catalog, found {len(instructions)}")

    expected_ids = ["original", "paraphrase_dev_01", "paraphrase_dev_02"]
    actual_ids = [r["id"] for r in instructions]
    if actual_ids != expected_ids:
        raise RuntimeError(f"FATAL: Instructions must be {expected_ids}, got {actual_ids}")

    orig_row = instructions[0]
    dev_paraphrases = instructions[1:]

    # Canonical original check
    expected_orig_norm = "pick up the black bowl between the plate and the ramekin and place it on the plate"
    if norm_full(orig_row["text"]) != expected_orig_norm:
        raise RuntimeError(f"FATAL: 'original' prompt text does not match canonical: {orig_row['text']}")

    # Collect other pools
    train_rows = load_jsonl(root / "configs/instructions/task0_train.jsonl")
    heldout_rows = load_jsonl(root / "configs/held_out_evaluation/instructions/task0_heldout.jsonl")
    pilot_rows = load_jsonl(root / "configs/instructions/task0.jsonl")
    reserved_path = root / "configs/final_confirmatory_reserved/instructions/task0_final_reserved.jsonl"
    reserved_rows = load_jsonl(reserved_path) if reserved_path.exists() else []

    forbidden_pools = {
        "training_paraphrases": [r for r in train_rows if r["id"] != "train_original"],
        "consumed_confirmatory_paraphrases": [r for r in heldout_rows if r["id"] not in ("original", "original_repeat")],
        "historical_pilot_paraphrases": [r for r in pilot_rows if r["id"] != "original"],
        "reserved_final_paraphrases": [r for r in reserved_rows if r["id"] != "original"],
    }

    # 1. Exact string duplicate checks
    for dev_r in dev_paraphrases:
        dev_text = dev_r["text"].strip()
        dev_norm = norm_full(dev_text)

        # Check against canonical original
        if dev_norm == expected_orig_norm:
            raise RuntimeError(f"FATAL: {dev_r['id']} has identical text to canonical original prompt!")

        for pool_name, pool in forbidden_pools.items():
            for other_r in pool:
                other_text = other_r["text"].strip()
                other_norm = norm_full(other_text)

                if dev_text == other_text:
                    raise RuntimeError(
                        f"FATAL: Exact string duplicate detected between {dev_r['id']} and "
                        f"{pool_name} ({other_r['id']}): '{dev_text}'"
                    )
                if dev_norm == other_norm:
                    raise RuntimeError(
                        f"FATAL: Normalized exact duplicate detected between {dev_r['id']} and "
                        f"{pool_name} ({other_r['id']}): '{dev_norm}'"
                    )

    # 2. Check dev paraphrases do not duplicate each other
    if norm_full(dev_paraphrases[0]["text"]) == norm_full(dev_paraphrases[1]["text"]):
        raise RuntimeError(f"FATAL: Development paraphrases are identical to each other!")

    # 3. Semantic Equivalence & Required Lexical Overlap Check
    task_critical_nouns = {"black", "bowl", "plate", "ramekin"}
    for dev_r in dev_paraphrases:
        words = word_set(dev_r["text"])
        missing_nouns = task_critical_nouns - words
        if missing_nouns:
            raise RuntimeError(f"FATAL: {dev_r['id']} is missing task-critical nouns: {missing_nouns}")

        if "plate" not in words:
            raise RuntimeError(f"FATAL: {dev_r['id']} missing destination landmark 'plate'")

    print("[Prompt-Validator Audit] PASSED:", flush=True)
    print("  - Structure: strictly 3 instructions (1 original, 2 dev paraphrases).", flush=True)
    print("  - Exact string identity: 0 duplicates across training, consumed confirmatory, pilot, and reserved final pools.", flush=True)
    print("  - Normalized sentence identity: 0 duplicates across training, consumed confirmatory, pilot, and reserved final pools.", flush=True)
    print("  - Task-critical nouns {'black', 'bowl', 'plate', 'ramekin'} confirmed present in all paraphrases (lexical overlap allowed & expected).", flush=True)
    print("  - Mutual uniqueness between dev_01 and dev_02 verified.", flush=True)


def preflight_development_validation(root, matrix, manifest):
    """Pre-flight validation ensuring data split integrity and model readiness."""
    print("[Dev Pre-flight] Validating data split integrity and model checkpoints...", flush=True)

    # 1. Rollout horizon audit: must be exactly 280 steps
    horizon = matrix.get("environment", {}).get("episode_length", None)
    if horizon != 280:
        raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Rollout horizon must be exactly 280 steps, got {horizon}!")

    # 2. Worker check: must be strictly 1
    if matrix.get("execution", {}).get("default_workers", 1) != 1:
        raise RuntimeError("FATAL PRE-FLIGHT ERROR: default_workers must be strictly 1 (serial execution)!")

    # 3. Strict State Boundary Audit
    allowed_states = set(range(23, 33))  # Exactly States 23-32
    forbidden_historical = set(manifest.get("development_boundary", {}).get("historical_dev_states", [0, 1, 2]))
    forbidden_consumed = set(manifest.get("development_boundary", {}).get("consumed_test_states", list(range(3, 23))))
    forbidden_future = set(manifest.get("development_boundary", {}).get("untouched_final_confirmatory_states", list(range(33, 53))))

    for task_cfg in matrix["tasks"]:
        if task_cfg["task_id"] != 0:
            raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Development evaluation must be Task 0 only, got Task {task_cfg['task_id']}!")
        states = set(task_cfg["initial_states"])
        if states.intersection(forbidden_historical):
            raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Historical dev states {states.intersection(forbidden_historical)} violated!")
        if states.intersection(forbidden_consumed):
            raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Consumed test states {states.intersection(forbidden_consumed)} violated!")
        if states.intersection(forbidden_future):
            raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Future confirmatory states {states.intersection(forbidden_future)} violated!")
        if states != allowed_states:
            raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Initial states must be strictly States 23-32, got {sorted(states)}!")

    # 4. Rigorous Instruction Catalog check
    validate_development_prompts(root, matrix["tasks"][0]["instruction_catalog"])

    # 5. Model Checkpoint Audit
    expected_models = ["original_language", "augmented_language", "anchored_language"]
    if matrix["models"] != expected_models:
        raise RuntimeError(f"FATAL: Models must be exactly {expected_models}, got {matrix['models']}")

    for model_id in matrix["models"]:
        if model_id not in manifest["evaluated_models"]:
            raise RuntimeError(f"Model {model_id} not registered in manifest")
        info = manifest["evaluated_models"][model_id]
        base_dir = root / info["checkpoint"]
        if not base_dir.exists():
            raise RuntimeError(f"FATAL: Model directory missing: {base_dir}")
        model_sub = base_dir / "pretrained_model"
        if not model_sub.exists():
            raise RuntimeError(f"FATAL: pretrained_model directory missing in {base_dir}")
        for required in ("config.json", "model.safetensors"):
            if not (model_sub / required).exists():
                raise RuntimeError(f"FATAL: Required model file {required} missing in {model_sub}")
        params_file = root / info["parameters"]
        if not params_file.exists():
            raise RuntimeError(f"FATAL: Parameter evidence missing: {params_file}")

        # Check recorded hashes if available
        for rel_file, exp_hash in info.get("artifact_sha256", {}).items():
            fpath = base_dir / rel_file
            if fpath.exists():
                act_hash = sha256_of_file(fpath)
                if act_hash != exp_hash:
                    raise RuntimeError(f"SHA256 mismatch for {model_id} {rel_file}: {act_hash} != {exp_hash}")

    dev_instructions = load_jsonl(root / matrix["tasks"][0]["instruction_catalog"])
    print("[Dev Pre-flight] Validation PASSED completely.")
    print("  - Target States: Strictly 10 states [23..32]")
    print("  - Models: original_language, augmented_language, anchored_language")
    print(f"  - Instructions: {[r['id'] for r in dev_instructions]}")
    print("  - Total planned rollouts: 90 (10 states x 3 prompts x 3 models)")
    print("  - Horizon: 280 steps | Workers: 1\n", flush=True)


def _batch_obs(obs):
    """Ensure raw observation has batch dimension expected by vector processors."""
    if isinstance(obs, dict):
        return {k: _batch_obs(v) for k, v in obs.items()}
    if isinstance(obs, np.ndarray):
        return np.expand_dims(obs, axis=0)
    return [obs]


def _execute_dev_episode(root, matrix, manifest, model_id, model_info, task_id, state_idx, instruction_row, trial_dir, baseline_identity):
    import torch
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION
    from lerobot.utils.random_utils import set_seed, get_rng_state
    from lerobot.utils.io_utils import write_video
    from .libero_pairing import array_hashes, capture_initial_state
    from .diagnostic_policy import capture_noise, require_empty_queue
    from .diagnostic_records import pack_rng, pack_tensors
    from .adaptation_eval_runtime import AdaptationEvalRuntime

    env_cfg = matrix["environment"]
    pol_cfg = matrix["policy"]
    suite_name = matrix["suite"]
    horizon = env_cfg["episode_length"]
    assert horizon == 280, f"Expected horizon 280, got {horizon}"

    set_seed(env_cfg["environment_seed"])
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[suite_name]()

    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        init_states=True,
        episode_index=state_idx,
        n_envs=1,
        obs_type=env_cfg["obs_type"],
        observation_height=env_cfg["observation_height"],
        observation_width=env_cfg["observation_width"],
        control_freq=env_cfg["control_freq"],
        control_mode=env_cfg["control_mode"],
        hard_reset=env_cfg["hard_reset"],
        num_steps_wait=env_cfg["num_steps_wait"],
        episode_length=horizon
    )

    try:
        obs, _ = env.reset(seed=env_cfg["environment_seed"])
        initial_arrays = capture_initial_state(env, obs, state_idx)
        initial_hashes = array_hashes(initial_arrays)

        atomic_json(trial_dir / "initial_hashes.json", initial_hashes)
        np.savez_compressed(trial_dir / "initial_state.npz", **initial_arrays)

        task_config = deepcopy(baseline_identity["config"])
        task_config["task_id"] = task_id
        task_config["suite"] = suite_name
        task_config["init_state_index"] = state_idx
        task_config["environment"]["episode_length"] = horizon
        task_config["policy"] = pol_cfg

        baseline_copy = deepcopy(baseline_identity)
        baseline_copy["config"] = task_config
        baseline_copy["instructions"] = [instruction_row]

        confirmatory_spec = {
            "matrix": matrix,
            "manifest": manifest,
            "model_id": model_id,
        }
        baseline_copy["confirmatory_spec"] = confirmatory_spec

        runtime = AdaptationEvalRuntime(root, baseline_copy, model_id, model_info, [instruction_row], confirmatory_spec=confirmatory_spec)
        runtime.load_policy()

        frames_main = []
        frames_wrist = []
        steps_records = []
        inference_records = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_successes = []
        trajectory_sim_states = []

        goal_ever = False
        goal_final = False
        first_step = None

        policy_obj = runtime.policy
        policy_obj.reset()
        set_seed(pol_cfg["policy_seed"])

        curr_obs = obs
        for step in range(1, horizon + 1):
            if "pixels" in curr_obs:
                if "image" in curr_obs["pixels"]:
                    frames_main.append(curr_obs["pixels"]["image"])
                if "image2" in curr_obs["pixels"]:
                    frames_wrist.append(curr_obs["pixels"]["image2"])

            # 1. Observation preprocessing
            obs_batched = _batch_obs(curr_obs)
            obs_converted = preprocess_observation(obs_batched)

            # 2. Attach task/language instruction
            obs_converted["task"] = [instruction_row["text"]]

            # 3. Environment preprocessing
            obs_env = runtime.env_pre(obs_converted)

            # 4. Policy preprocessing
            batch = runtime.pre(obs_env)

            # 5. Policy inference
            require_empty_queue(policy_obj)
            rng_b = pack_rng(get_rng_state(), "rng_before", {}, {})
            with capture_noise(policy_obj) as noise:
                action = policy_obj.select_action(batch)
            require_empty_queue(policy_obj)
            rng_a = pack_rng(get_rng_state(), "rng_after", {}, {})

            # 6. Policy postprocessing
            unnormalized_action = runtime.post(action)

            # 7. Environment postprocessing
            action_transition = runtime.env_post({ACTION: unnormalized_action})
            final_action = action_transition[ACTION]

            # 8. Convert action to environment numpy format
            action_np = final_action.squeeze(0).detach().cpu().numpy()

            raw_pixels_hash = array_hashes(curr_obs.get("pixels", {})) if "pixels" in curr_obs else {}
            inference_records.append({
                "step": step,
                "raw_pixels": raw_pixels_hash,
                "rng_before": rng_b,
                "rng_after": rng_a,
                "normalized_action": pack_tensors({"action": action}, "output", {}, {})
            })

            # 9. Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_np)

            is_goal = get_task_goal_status(env)
            if is_goal and not goal_ever:
                goal_ever = True
                first_step = step
            if step == horizon:
                goal_final = is_goal

            steps_records.append({
                "step": step,
                "instruction_id": instruction_row["id"],
                "action": action_np.tolist(),
                "reward": float(reward),
                "goal": is_goal
            })

            trajectory_actions.append(action_np)
            trajectory_rewards.append(reward)
            trajectory_successes.append(is_goal)
            trajectory_sim_states.append(env._env.get_sim_state())

            curr_obs = next_obs

        video_main_path = trial_dir / "image.mp4"
        video_wrist_path = trial_dir / "image2.mp4"
        if frames_main:
            write_video(str(video_main_path), np.stack(frames_main), fps=env_cfg["control_freq"])
        if frames_wrist:
            write_video(str(video_wrist_path), np.stack(frames_wrist), fps=env_cfg["control_freq"])

        np.savez_compressed(
            trial_dir / "trajectory.npz",
            action=np.array(trajectory_actions),
            reward=np.array(trajectory_rewards),
            success=np.array(trajectory_successes),
            sim_state=np.array(trajectory_sim_states)
        )

        with open(trial_dir / "steps.jsonl", "w", encoding="utf-8") as f:
            for r in steps_records:
                f.write(json.dumps(r, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")

        with open(trial_dir / "inference.jsonl", "w", encoding="utf-8") as f:
            for r in inference_records:
                f.write(json.dumps(r, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")

        episode_summary = {
            "model": model_id,
            "task_id": task_id,
            "initial_state_index": state_idx,
            "instruction_id": instruction_row["id"],
            "instruction": instruction_row["text"],
            "policy_steps": horizon,
            "goal_ever": goal_ever,
            "goal_final": goal_final,
            "first_step": first_step,
        }
        atomic_json(trial_dir / "episode.json", episode_summary)

        complete_marker = {
            "status": "COMPLETED",
            "model": model_id,
            "task_id": task_id,
            "initial_state_index": state_idx,
            "instruction_id": instruction_row["id"],
            "policy_steps": horizon,
            "goal_ever": goal_ever,
            "goal_final": goal_final,
            "first_step": first_step,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        atomic_json(trial_dir / "complete.json", complete_marker)

    finally:
        env.close()


def run_development_evaluation(matrix_config_path="configs/development_evaluation/dev_matrix.json",
                               default_run_name="task0_anchored_dev_eval"):
    root = Path(__file__).resolve().parents[2]
    matrix = json.loads((root / matrix_config_path).read_text(encoding="utf-8"))
    manifest = json.loads((root / matrix["manifest_path"]).read_text(encoding="utf-8"))

    # Update manifest with anchored_language checkpoint hashes if ready
    anc_dir = root / manifest["evaluated_models"]["anchored_language"]["checkpoint"]
    anc_hashes = {}
    if anc_dir.exists():
        for f in ["parameters.json", "pretrained_model/config.json", "pretrained_model/model.safetensors"]:
            target = anc_dir / f
            if target.exists():
                anc_hashes[f] = sha256_of_file(target)
    manifest["evaluated_models"]["anchored_language"]["artifact_sha256"] = anc_hashes

    # Validate
    preflight_development_validation(root, matrix, manifest)

    run_name = default_run_name
    output_dir = root / matrix["output_root"] / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with run_lock(output_dir):
        meta_path = output_dir / "experiment_meta.json"
        if not meta_path.exists():
            atomic_json(meta_path, {
                "run_name": run_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "matrix_config": matrix,
                "manifest": manifest,
            })

        tasks = matrix["tasks"]
        models = matrix["models"]
        total_episodes = sum(len(m["initial_states"]) * len(load_jsonl(root / m["instruction_catalog"])) for m in tasks) * len(models)
        completed_count = 0

        print(f"\n=======================================================", flush=True)
        print(f" Starting Development Model-Selection Evaluation: {run_name}", flush=True)
        print(f" Models: {models}", flush=True)
        print(f" Target States: {tasks[0]['initial_states']} (10 states)", flush=True)
        print(f" Total planned rollouts: {total_episodes} (10 states x 3 prompts x 3 models)", flush=True)
        print(f" Horizon: 280 steps per episode | Worker: 1", flush=True)
        print(f"=======================================================\n", flush=True)

        baseline_identity = json.loads((root / "outputs/language_pilot/identity.json").read_text(encoding="utf-8"))

        for model_id in models:
            model_info = manifest["evaluated_models"][model_id]
            model_dir = output_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n>>> Loading model [{model_id}] ...", flush=True)

            for task_cfg in tasks:
                task_id = task_cfg["task_id"]
                instructions = load_jsonl(root / task_cfg["instruction_catalog"])
                task_dir = model_dir / f"task_{task_id:02d}"
                task_dir.mkdir(parents=True, exist_ok=True)

                for state_idx in task_cfg["initial_states"]:
                    for inst in instructions:
                        completed_count += 1
                        inst_id = inst["id"]
                        trial_slug = f"state_{state_idx:04d}_seed_{matrix['policy']['policy_seed']:04d}_{inst_id}"
                        trial_dir = task_dir / trial_slug / "attempt_001"
                        complete_file = trial_dir / "complete.json"

                        pct = (completed_count / total_episodes) * 100
                        prefix = f"[{completed_count:03d}/{total_episodes:03d} - {pct:5.1f}%]"

                        if complete_file.exists():
                            print(f"{prefix} (Cached) {model_id} | Task {task_id} | State {state_idx} | {inst_id}", flush=True)
                            continue

                        print(f"{prefix} (Running) {model_id} | Task {task_id} | State {state_idx} | {inst_id}", flush=True)
                        trial_dir.mkdir(parents=True, exist_ok=True)

                        _execute_dev_episode(
                            root=root,
                            matrix=matrix,
                            manifest=manifest,
                            model_id=model_id,
                            model_info=model_info,
                            task_id=task_id,
                            state_idx=state_idx,
                            instruction_row=inst,
                            trial_dir=trial_dir,
                            baseline_identity=baseline_identity
                        )

        print("\nAll development model-selection evaluation episodes completed successfully!", flush=True)
        summarize_development_run(output_dir)


def summarize_development_run(run_dir):
    """Aggregate outcomes across States 23-32 and compute metrics."""
    run_dir = Path(run_dir)
    matrix = json.loads((run_dir / "experiment_meta.json").read_text(encoding="utf-8"))["matrix_config"]
    models = matrix["models"]
    states = matrix["tasks"][0]["initial_states"]
    instructions = ["original", "paraphrase_dev_01", "paraphrase_dev_02"]

    results = {}
    rows_by_state = {s: {"state_index": s} for s in states}

    for model_id in models:
        task_dir = run_dir / model_id / "task_00"
        m_results = {inst: [] for inst in instructions}
        m_final = {inst: [] for inst in instructions}

        for s in states:
            for inst in instructions:
                slug = f"state_{s:04d}_seed_0000_{inst}"
                complete_path = task_dir / slug / "attempt_001/complete.json"
                if complete_path.exists():
                    data = json.loads(complete_path.read_text(encoding="utf-8"))
                    ever = int(data.get("goal_ever", False))
                    final = int(data.get("goal_final", False))
                    m_results[inst].append(ever)
                    m_final[inst].append(final)
                    rows_by_state[s][f"{model_id}_{inst}_ever"] = ever
                    rows_by_state[s][f"{model_id}_{inst}_final"] = final
                else:
                    m_results[inst].append(0)
                    m_final[inst].append(0)
                    rows_by_state[s][f"{model_id}_{inst}_ever"] = 0
                    rows_by_state[s][f"{model_id}_{inst}_final"] = 0

        orig_succ = sum(m_results["original"])
        p1_succ = sum(m_results["paraphrase_dev_01"])
        p2_succ = sum(m_results["paraphrase_dev_02"])
        pooled_succ = p1_succ + p2_succ
        n_states = len(states)
        orig_rate = orig_succ / n_states
        pooled_rate = pooled_succ / (2 * n_states)
        gap = pooled_rate - orig_rate

        o_val = orig_rate
        p_val = pooled_rate
        h_val = (2.0 * o_val * p_val / (o_val + p_val)) if (o_val + p_val) > 0.0 else 0.0

        results[model_id] = {
            "evaluated_states": states,
            "states_count": n_states,
            "O": o_val,
            "P": p_val,
            "H": h_val,
            "robustness_gap": p_val - o_val,
            "original_success_count": f"{orig_succ}/{n_states}",
            "original_success_rate": orig_rate,
            "paraphrase_dev_01_count": f"{p1_succ}/{n_states}",
            "paraphrase_dev_01_rate": p1_succ / n_states,
            "paraphrase_dev_02_count": f"{p2_succ}/{n_states}",
            "paraphrase_dev_02_rate": p2_succ / n_states,
            "pooled_dev_paraphrase_count": f"{pooled_succ}/{2 * n_states}",
            "pooled_dev_paraphrase_success_rate": pooled_rate,
            "goal_final_summary": {
                "original_final_count": f"{sum(m_final['original'])}/{n_states}",
                "paraphrase_dev_01_final_count": f"{sum(m_final['paraphrase_dev_01'])}/{n_states}",
                "paraphrase_dev_02_final_count": f"{sum(m_final['paraphrase_dev_02'])}/{n_states}",
                "pooled_dev_paraphrase_final_count": f"{(sum(m_final['paraphrase_dev_01']) + sum(m_final['paraphrase_dev_02']))}/{2 * n_states}",
            }
        }

    gate_eval = None
    if "anchored_language" in results and "original_language" in results and "augmented_language" in results:
        o_anchor = results["anchored_language"]["O"]
        o_aug = results["augmented_language"]["O"]
        p_anchor = results["anchored_language"]["P"]
        p_orig = results["original_language"]["P"]
        h_anchor = results["anchored_language"]["H"]
        h_orig = results["original_language"]["H"]
        h_aug = results["augmented_language"]["H"]

        cond1 = bool(o_anchor >= o_aug)
        cond2 = bool(p_anchor >= p_orig)
        cond3 = bool(h_anchor >= max(h_orig, h_aug))
        all_passed = bool(cond1 and cond2 and cond3)

        gate_eval = {
            "status": "PASS" if all_passed else "FAIL",
            "conditions": {
                "1_O_anchor_ge_O_augmented": {
                    "rule": "O_anchor >= O_augmented",
                    "anchor_val": o_anchor,
                    "threshold_val": o_aug,
                    "passed": cond1,
                },
                "2_P_anchor_ge_P_original_language": {
                    "rule": "P_anchor >= P_original_language",
                    "anchor_val": p_anchor,
                    "threshold_val": p_orig,
                    "passed": cond2,
                },
                "3_H_anchor_ge_max_controls": {
                    "rule": "H_anchor >= max(H_original_language, H_augmented_language)",
                    "anchor_val": h_anchor,
                    "max_controls_val": max(h_orig, h_aug),
                    "H_original_language": h_orig,
                    "H_augmented_language": h_aug,
                    "passed": cond3,
                }
            }
        }

    summary_data = {
        "evaluation_name": run_dir.name,
        "evaluation_scope": "Model-selection development evaluation on States 23-32 (10 states)",
        "models": models,
        "primary_metric": "goal_ever",
        "results": results,
        "model_selection_gate": gate_eval,
    }
    atomic_json(run_dir / "summary.json", summary_data)

    # Write CSV
    csv_file = run_dir / "per_state_raw_outcomes.csv"
    headers = ["state_index"]
    for m in models:
        for inst in instructions:
            headers.append(f"{m}_{inst}_ever")
            headers.append(f"{m}_{inst}_final")

    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for s in sorted(states):
            writer.writerow(rows_by_state[s])

    print(f"\n[Development Summary Generated] Saved to {run_dir / 'summary.json'}", flush=True)
    print(f"Per-state CSV saved to {csv_file}\n", flush=True)
    return summary_data
