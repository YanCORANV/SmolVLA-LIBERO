"""Verify and summarize the single acceptance rollout; never rerun evaluation."""

import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess

import yaml


def run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def probe(path):
    return json.loads(run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames,duration",
        "-of", "json", str(path),
    ]))["streams"][0]


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "outputs/first_rollout"
    destination = root / "results/first_rollout"
    print("[1/3] Validate the saved single-episode result", flush=True)
    if int((source / "exit_code.txt").read_text()) != 0:
        raise RuntimeError("Evaluation did not exit successfully")
    info = json.loads((source / "eval_info.json").read_text())
    config = yaml.safe_load((source / "run_config.yaml").read_text())
    checkpoint = json.loads((source / "checkpoint.json").read_text())
    if info["overall"]["n_episodes"] != 1 or len(info["per_task"]) != 1:
        raise RuntimeError("This acceptance summarizer expects exactly one task and episode")
    task = info["per_task"][0]
    successes = task["metrics"]["successes"]
    if len(successes) != 1 or not isinstance(successes[0], bool):
        raise RuntimeError("Unexpected success field")
    if task["task_group"] != config["env"]["task"] or [task["task_id"]] != config["env"]["task_ids"]:
        raise RuntimeError("Result does not match saved task configuration")
    video = root / task["metrics"]["video_paths"][0]
    stream = probe(video)
    frames = int(stream["nb_read_frames"])
    if frames < 2:
        raise RuntimeError("Video contains fewer than two frames")
    run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"])

    print("[2/3] Preserve original video and create presentation artifacts", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    # The installed LiberoEnv config passes fps to control_freq; its default is 20.
    control_hz = config["env"].get("fps", 20)
    encoded_fps = float(Fraction(stream["avg_frame_rate"]))
    playback = video.with_name("eval_episode_0_playback_20fps.mp4")
    run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vf", f"setpts={encoded_fps / control_hz}*PTS", "-r", str(control_hz),
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(playback)])
    corrected_stream = probe(playback)
    if int(corrected_stream["nb_read_frames"]) != frames:
        raise RuntimeError("Presentation copy changed the video frame count")
    indices = [round(i * (frames - 1) / 5) for i in range(6)]
    selection = "+".join(f"eq(n\\,{i})" for i in indices)
    run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vf", f"select={selection},tile=3x2:padding=8:margin=8",
         "-frames:v", "1", str(destination / "keyframes.png")])

    print("[3/3] Export CSV/JSON with timing and memory measurement scopes", flush=True)
    with (source / "gpu_memory.csv").open(newline="") as handle:
        samples = list(csv.DictReader(handle, skipinitialspace=True))
    gpu_memory = [float(row["memory.used [MiB]"].split()[0]) for row in samples]
    resources = (source / "process_resources.txt").read_text()
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (.+)", resources).group(1)
    wall_seconds = 0.0
    for part in wall.split(":"):
        wall_seconds = wall_seconds * 60 + float(part)
    peak_rss_kib = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", resources).group(1))
    row = {
        "run_id": "first_rollout", "suite": task["task_group"], "task_id": task["task_id"],
        "seed": config["seed"], "language_condition": "original", "episodes": 1,
        "success": successes[0], "sum_reward": task["metrics"]["sum_rewards"][0],
        "checkpoint_repo": checkpoint["repo_id"], "checkpoint_revision": checkpoint["revision"],
        "evaluation_seconds": info["overall"]["eval_s"], "command_wall_seconds": wall_seconds,
        "process_peak_rss_mib": peak_rss_kib / 1024,
        "sampled_device_peak_vram_mib": max(gpu_memory), "gpu_samples": len(samples),
        "video_frames": frames, "video_width": stream["width"], "video_height": stream["height"],
        "original_video_fps": encoded_fps, "control_hz": control_hz,
        "playback_video_fps": float(Fraction(corrected_stream["avg_frame_rate"])),
        "original_video": video.relative_to(root).as_posix(),
        "playback_video": playback.relative_to(root).as_posix(),
    }
    report = {
        "result": row, "original_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "video_decoded_without_error": True,
        "keyframe_indices": indices,
        "memory_scope": "Device-wide one-second samples including other GPU users; not an exact per-process peak",
        "result_scope": "One acceptance episode, not a benchmark success-rate estimate",
        "video_timing_note": "Upstream video metadata was 80 FPS while control was 20 Hz; a derived playback copy is retimed without changing the original or episode outcome",
        "initial_state_note": "init_states=true and a single environment were configured; simulator state and policy RNG were not serialized during this acceptance run",
        "action_log_note": "Per-step actions were not exported by the upstream CLI in this run; whole-trajectory finiteness was not independently checked",
        "raw_eval_info": info,
    }
    (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (destination / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2), flush=True)
    print("PASS: original video decodes; presentation copy preserves frame count; result tables saved.", flush=True)


if __name__ == "__main__":
    main()
