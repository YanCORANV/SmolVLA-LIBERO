"""Audit saved pilot evidence and extract fixed-time frames; never run the policy.

Episode checkpoints make interrupted extraction resumable. Raw run artifacts remain
unchanged. Only reviewed small tables/figures should later be copied into results/.
"""

import json
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.protocol import Checkpoints, atomic_json, file_sha256, fingerprint, run_lock
from smolvla_grounding.rollout_records import summarize_goals


def first_difference(a, b, threshold=0):
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    per_step = difference.reshape(len(a), -1).max(axis=1)
    indices = np.flatnonzero(per_step > threshold)
    return {"first_array_index": int(indices[0]) if len(indices) else None,
            "max_absolute_difference": float(per_step.max())}


def main():
    source = ROOT / "outputs/language_pilot"
    destination = ROOT / "outputs/language_pilot_review"
    summary = json.loads((source / "summary.json").read_text())
    identity = json.loads((source / "identity.json").read_text())
    if not summary["all_trials_complete"] or len(summary["episodes"]) != 6:
        raise RuntimeError("Expected the completed six-condition pilot")
    source_fingerprint = fingerprint(identity)
    with run_lock(destination):
        checkpoints = Checkpoints(destination, {
            "run_fingerprint": source_fingerprint, "summary_sha256": file_sha256(source / "summary.json"),
            "analysis_source_sha256": file_sha256(Path(__file__)),
        })
        trajectories, rngs, initials, audited = {}, {}, {}, []
        for index, episode in enumerate(summary["episodes"], 1):
            name = episode["instruction_id"]
            print(f"[{index}/6] Verify logs, arrays and video for {name}", flush=True)
            marker = json.loads((source / name / "complete.json").read_text())
            if marker["fingerprint"] != source_fingerprint or marker["result"] != episode:
                raise RuntimeError("Run marker/result mismatch")
            for relative, expected in marker["artifacts"].items():
                if file_sha256(source / relative) != expected:
                    raise RuntimeError(f"Artifact changed: {relative}")
            attempt = (source / episode["trajectory"]).parent
            records = [json.loads(line) for line in (attempt / "steps.jsonl").read_text().splitlines()]
            with np.load(attempt / "trajectory.npz", allow_pickle=False) as saved:
                arrays = {k: saved[k] for k in saved.files}
            with np.load(attempt / "initial_rng.npz", allow_pickle=False) as saved:
                rngs[name] = {k: saved[k] for k in saved.files}
            with np.load(attempt / "initial_state.npz", allow_pickle=False) as saved:
                initials[name] = {k: saved[k] for k in saved.files}
            if len(records) != 280 or arrays["action"].shape != (280, 7) or len(arrays["sim_state"]) != 281:
                raise RuntimeError("Unexpected trajectory lengths")
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise RuntimeError("Nonfinite trajectory")
            if not np.array_equal(arrays["action"], np.asarray([r["action"] for r in records], dtype=arrays["action"].dtype)):
                raise RuntimeError("Action log/array mismatch")
            for step, record in enumerate(records, 1):
                if record["step"] != step or record["instruction"] != episode["instruction"]:
                    raise RuntimeError("Step indexing or language mismatch")
            goals = summarize_goals([r["goals"] for r in records], identity["config"]["goals"])
            if goals != episode["goals"]:
                raise RuntimeError("Goal summary mismatch")
            for goal in goals:
                if not np.array_equal(arrays["goal_" + goal], [r["goals"][goal] for r in records]):
                    raise RuntimeError("Goal log/array mismatch")
            trajectories[name] = arrays
            completed = checkpoints.completed(name)
            if completed is None:
                output = checkpoints.new_attempt(name)
                selected = [0, 45, 78, 140, 210, 280]
                artifacts, video_info = [], {}
                for camera, video_key in (("main", "video_main"), ("wrist", "video_wrist")):
                    frames = {}
                    with av.open(str(source / episode[video_key])) as video:
                        fps = float(video.streams.video[0].average_rate)
                        count = 0
                        for count, frame in enumerate(video.decode(video=0), 1):
                            if count - 1 in selected:
                                frames[count - 1] = frame.to_image()
                    if count != 281 or fps != 20 or len(frames) != len(selected):
                        raise RuntimeError("Invalid video length/rate")
                    sheet = Image.new("RGB", (3 * 256, 2 * 282 + 26), "white")
                    draw = ImageDraw.Draw(sheet)
                    draw.text((8, 6), f"{name} / {camera} / frames at fixed policy steps", fill="black")
                    for i, step in enumerate(selected):
                        x, y = (i % 3) * 256, (i // 3) * 282 + 26
                        draw.text((x + 8, y + 5), f"step {step} ({step / 20:.2f} s)", fill="black")
                        sheet.paste(frames[step], (x, y + 26))
                    sheet_path = output / f"{camera}_frames.png"
                    sheet.save(sheet_path)
                    artifacts.append(sheet_path)
                    video_info[camera] = {"frames": count, "fps": fps}
                completed = {"instruction_id": name, "goals": goals, "video_info": video_info,
                             "frame_sheets": [p.relative_to(destination).as_posix() for p in artifacts]}
                checkpoints.finish(name, completed, artifacts)
            audited.append(completed)

        print("[7/7] Locate repeated-original divergence and export audit", flush=True)
        original, repeat = trajectories["original"], trajectories["original_repeat"]
        rng_equal = all(rngs[n].keys() == rngs["original"].keys() and
                        all(np.array_equal(v, rngs["original"][k]) for k, v in rngs[n].items()) for n in rngs)
        initial_equal = all(initials[n].keys() == initials["original"].keys() and
                            all(np.array_equal(v, initials["original"][k]) for k, v in initials[n].items()) for n in initials)
        action_diff = first_difference(original["action"], repeat["action"])
        first_index = action_diff["first_array_index"]
        audit = {"run_fingerprint": source_fingerprint, "artifact_integrity_passed": True,
                 "all_initial_arrays_equal": initial_equal, "all_initial_rng_arrays_equal": rng_equal,
                 "episode_audits": audited, "repeat_control": {
                     "first_action_difference_step": None if first_index is None else first_index + 1,
                     "max_absolute_action_difference": action_diff["max_absolute_difference"],
                     "difference_at_first_diverging_action": None if first_index is None else
                         float(np.abs(original["action"][first_index] - repeat["action"][first_index]).max()),
                     "sim_state": first_difference(original["sim_state"], repeat["sim_state"]),
                     "goal_disagreement_steps": {g: (np.flatnonzero(original["goal_" + g] != repeat["goal_" + g]) + 1).tolist()
                                                 for g in identity["config"]["goals"]}},
                 "first_action_differences_from_original": {
                     n: float(np.abs(a["action"][0] - original["action"][0]).max()) for n, a in trajectories.items()},
                 "total_rollout_seconds": sum(e["rollout_seconds"] for e in summary["episodes"]),
                 "backend": identity["backend"],
                 "diagnostic_limitations": [
                     "Per-step policy input tensors, RNG and sampled flow noise were not recorded.",
                     "Equal physics arrays before a differing action do not prove equal rendered policy inputs.",
                     "Decoded lossy videos cannot establish exact raw observation equality.",
                     "Numerical/kernel nondeterminism, observations and unrecorded internal state remain candidate causes.",
                     "Two successful original trials do not estimate outcome variability across seeds/states.",
                 ]}
        if not initial_equal or not rng_equal:
            raise RuntimeError("Saved initial array/RNG pairing failed")
        atomic_json(destination / "audit.json", audit)
        print(json.dumps(audit["repeat_control"], indent=2), flush=True)
        print("PASS: saved artifacts audited; review fixed-time frames before labeling failures.", flush=True)


if __name__ == "__main__":
    main()
