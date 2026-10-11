"""Publish deterministic, restart-safe video evidence for the frozen pilot cases."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.protocol import file_sha256

RESULT = ROOT / "results/multitask_language_pilot"
MATRIX = json.loads((ROOT / "configs/multitask_evaluation/language_pilot_matrix.json").read_text(encoding="utf-8"))
RUN = json.loads((ROOT / MATRIX["output_dir"] / "experiment_meta.json").read_text(encoding="utf-8"))
PROMPTS = ("original", "paraphrase_01", "paraphrase_02")
COLUMNS = (
    "case_id", "selection", "model", "experiment_identity", "suite", "task_id", "state",
    "canonical_prompt", "canonical_text", "paraphrase_prompt", "paraphrase_text",
    "canonical_goal_ever", "paraphrase_goal_ever", "canonical_goal_final", "paraphrase_goal_final",
    "outcome_transition", "step1_action_l2_normalized", "early_action_l2_normalized",
    "eef_at_step_10_cm", "failure_category", "mechanism",
    "canonical_evidence_dir", "paraphrase_evidence_dir",
    "canonical_video_sha256", "paraphrase_video_sha256", "frame_indices",
    "case_sheet", "case_sheet_sha256",
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verified_video(row):
    folder = (ROOT / row["evidence_dir"]).resolve()
    if not folder.is_relative_to((ROOT / MATRIX["output_dir"]).resolve()):
        raise RuntimeError("Evidence path leaves the frozen pilot output")
    commit_path = folder / "complete.json"
    if file_sha256(commit_path) != row["complete_sha256"]:
        raise RuntimeError(f"Committed outcome hash differs: {commit_path}")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if (commit["model"], commit["suite"], commit["task_id"], commit["state"],
            commit["prompt_id"], str(commit["goal_ever"]), str(commit["goal_final"])) != (
            row["model"], row["suite"], int(row["task_id"]), int(row["state"]),
            row["prompt"], row["goal_ever"], row["goal_final"]):
        raise RuntimeError(f"Committed outcome differs from public table: {commit_path}")
    video = folder / "image.mp4"
    digest = file_sha256(video)
    if digest != commit["artifacts"]["image.mp4"]:
        raise RuntimeError(f"Committed video hash differs: {video}")
    return video, digest


def frames(video, horizon):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count != horizon:
        capture.release()
        raise RuntimeError(f"Video has {count} frames, expected {horizon}: {video}")
    indices = [0, horizon // 4, horizon // 2, 3 * horizon // 4, horizon - 1]
    images = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Cannot decode frame {index}: {video}")
        images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return indices, images


def make_sheet(path, labels, videos, horizon):
    sheet = Image.new("RGB", (160 + 5 * 256, 2 * 276), "white")
    draw = ImageDraw.Draw(sheet)
    indices = None
    for row, (label, video) in enumerate(zip(labels, videos)):
        selected, images = frames(video, horizon)
        if indices is not None and indices != selected:
            raise RuntimeError("Paired keyframe indices differ")
        indices = selected
        y = row * 276
        draw.text((8, y + 12), label, fill="black")
        for col, (index, picture) in enumerate(zip(selected, images)):
            x = 160 + col * 256
            sheet.paste(picture.resize((256, 256)), (x, y))
            draw.text((x + 5, y + 236), str(index), fill="white", stroke_width=1,
                      stroke_fill="black")
    temporary = path.with_name(path.name + ".tmp")
    sheet.save(temporary, format="PNG")
    os.replace(temporary, path)
    return indices


def main():
    rollouts = read_csv(RESULT / "per_rollout.csv")
    pairs = read_csv(RESULT / "pair_level.csv")
    if len(rollouts) != 75 or len(pairs) != 50:
        raise RuntimeError("Expected 75 rollouts and 50 pairs")
    by_id = {(r["suite"], int(r["task_id"]), int(r["state"]), r["prompt"]): r
             for r in rollouts}
    if len(by_id) != 75:
        raise RuntimeError("Duplicate public rollout identity")
    chosen = []
    for task in MATRIX["tasks"]:
        candidates = [p for p in pairs if p["suite"] == task["suite"]
                      and int(p["task_id"]) == task["task_id"]]
        seen_states = set()
        for pair in candidates:
            if pair["outcome_transition"] != "canonical_success_paraphrase_failure":
                continue
            state = int(pair["state"])
            if state in seen_states:
                continue
            chosen.append(("failure", pair))
            seen_states.add(state)
            if len(seen_states) == 2:
                break
        if len(seen_states) != 2:
            raise RuntimeError(f"Fewer than two distinct failure states for {task['suite']}/{task['task_id']}")
    for task in MATRIX["tasks"]:
        contrast = next((p for p in pairs if p["suite"] == task["suite"]
                         and int(p["task_id"]) == task["task_id"]
                         and p["outcome_transition"] == "both_succeed"), None)
        if contrast is not None:
            chosen.append(("success_contrast", contrast))
    if len(chosen) < 10:
        raise RuntimeError("Fewer than 10 auditable comparison cases")
    prior = {r["case_id"]: r for r in read_csv(RESULT / "case_index.csv")} if (
        RESULT / "case_index.csv").exists() else {}
    case_dir = RESULT / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ordinal, (selection, pair) in enumerate(chosen, 1):
        suite, task_id, state, prompt = pair["suite"], int(pair["task_id"]), int(pair["state"]), pair["prompt"]
        task = next(t for t in MATRIX["tasks"] if (t["suite"], t["task_id"]) == (suite, task_id))
        catalog = {p["id"]: p["text"] for p in (
            json.loads(line) for line in (ROOT / task["prompt_catalog"]).read_text(encoding="utf-8").splitlines())}
        canonical = by_id[(suite, task_id, state, "original")]
        variant = by_id[(suite, task_id, state, prompt)]
        canonical_video, canonical_digest = verified_video(canonical)
        variant_video, variant_digest = verified_video(variant)
        case_id = f"C{ordinal:02d}"
        rel_sheet = f"results/multitask_language_pilot/cases/{case_id}.png"
        sheet = ROOT / rel_sheet
        previous = prior.get(case_id)
        can_skip = (previous is not None and previous["suite"] == suite
                    and int(previous["task_id"]) == task_id
                    and int(previous["state"]) == state
                    and previous["paraphrase_prompt"] == prompt
                    and previous["canonical_video_sha256"] == canonical_digest
                    and previous["paraphrase_video_sha256"] == variant_digest
                    and sheet.is_file() and file_sha256(sheet) == previous["case_sheet_sha256"])
        if can_skip:
            indices = [int(x) for x in previous["frame_indices"].split(";")]
            print(f"[Case {ordinal:02d}/{len(chosen)}] cached {suite}/{task_id} state={state} {prompt}", flush=True)
        else:
            indices = make_sheet(sheet, ("original", prompt),
                                 (canonical_video, variant_video), task["horizon"])
            print(f"[Case {ordinal:02d}/{len(chosen)}] rendered {suite}/{task_id} state={state} {prompt}", flush=True)
        rows.append({
            "case_id": case_id, "selection": selection, "model": "untuned_fp32",
            "experiment_identity": RUN["identity"], "suite": suite, "task_id": task_id,
            "state": state, "canonical_prompt": "original", "canonical_text": catalog["original"],
            "paraphrase_prompt": prompt, "paraphrase_text": catalog[prompt],
            "canonical_goal_ever": canonical["goal_ever"],
            "paraphrase_goal_ever": variant["goal_ever"],
            "canonical_goal_final": canonical["goal_final"],
            "paraphrase_goal_final": variant["goal_final"],
            "outcome_transition": pair["outcome_transition"],
            "step1_action_l2_normalized": pair["first_action_l2_normalized"],
            "early_action_l2_normalized": pair["early_action_l2_normalized"],
            "eef_at_step_10_cm": f"{float(pair['eef_at_step_10_m']) * 100:.8f}",
            "failure_category": "task_timeout" if selection == "failure" else "none_success",
            "mechanism": "unknown" if selection == "failure" else "not_applicable",
            "canonical_evidence_dir": canonical["evidence_dir"],
            "paraphrase_evidence_dir": variant["evidence_dir"],
            "canonical_video_sha256": canonical_digest,
            "paraphrase_video_sha256": variant_digest,
            "frame_indices": ";".join(str(x) for x in indices),
            "case_sheet": rel_sheet, "case_sheet_sha256": file_sha256(sheet),
        })
    destination = RESULT / "case_index.csv"
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    print(f"[Case evidence] {len(rows)} indexed comparisons; {sum(r['selection'] == 'failure' for r in rows)} failures", flush=True)


if __name__ == "__main__":
    main()
