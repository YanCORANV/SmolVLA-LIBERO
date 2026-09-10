"""Extract auditable qualitative evidence from the completed development run.

No model or simulator is loaded. Object displacement describes observed motion;
it is not an automatic grounding or grasp-success classifier.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.benchmark_records import BenchmarkJournal
from smolvla_grounding.protocol import atomic_json, file_sha256, run_lock


def main():
    import av
    import numpy as np
    from PIL import Image, ImageDraw

    source = ROOT / "outputs/adaptation_evaluation/runs/task0_adaptation_eval"
    directory = ROOT / "outputs/adaptation_review/task0_adaptation_eval"
    comparison = json.loads((source / "comparison.json").read_text())
    if not comparison["all_complete"]:
        raise RuntimeError("Complete the three-model evaluation first")
    # Review original-instruction regressions, paired B/C baseline versus tuning,
    # and the augmented model's newly successful original-instruction state.
    selected = [(m, 0, n) for m, names in (
        ("untuned_fp32", ("original", "paraphrase_01", "paraphrase_02")),
        ("original_language", ("original",)),
        ("augmented_language", ("original", "paraphrase_01", "paraphrase_02"))) for n in names]
    selected.append(("augmented_language", 1, "original"))
    frames_at = [0, 40, 80, 120, 160, 200, 240, 280]
    identity = {"comparison_sha256": file_sha256(source / "comparison.json"),
                "script_sha256": file_sha256(__file__), "selected": selected, "frames_at": frames_at}
    # JSON roundtrip keeps tuples comparable to a resumed saved identity.
    identity = json.loads(json.dumps(identity))
    with run_lock(directory):
        if (directory / "identity.json").exists():
            if input("Continue this video review and verify saved evidence? [y/N]: ").strip().lower() != "y":
                return
        journal = BenchmarkJournal(directory, identity)
        outcomes = []
        for i, (model, state, instruction) in enumerate(selected, 1):
            key = f"{model}_state_{state}_{instruction}"
            print(f"[{i}/{len(selected)}] {key}", flush=True)
            episodes = json.loads((source / model / "episodes.json").read_text())
            episode = next(e for e in episodes if (e["initial_state_index"], e["policy_seed"], e["instruction_id"]) == (state, 0, instruction))
            trial = source / model / Path(episode["trajectory"]).parent
            inputs = {str(p.relative_to(ROOT).as_posix()): file_sha256(p) for p in (
                trial / "steps.jsonl", trial / "image.mp4", trial / "image2.mp4")}
            saved = journal.completed(key)
            if saved is not None:
                if saved["input_hashes"] != inputs:
                    raise RuntimeError("Source video/log changed")
                outcomes.append(saved)
                continue
            attempt = journal.new_attempt(key)
            records = [json.loads(line) for line in (trial / "steps.jsonl").read_text().splitlines()]
            if len(records) != 280:
                raise RuntimeError("Incomplete trajectory logs")
            objects = {}
            for name in records[0]["object_positions"]:
                xyz = np.asarray([r["object_positions"][name] for r in records])
                objects[name] = {"max_z_rise_from_step_1_m": float((xyz[:, 2] - xyz[0, 2]).max()),
                    "peak_z_step": int(xyz[:, 2].argmax()) + 1,
                    "final_xy_displacement_from_step_1_m": float(np.linalg.norm(xyz[-1, :2] - xyz[0, :2])),
                    "final_xyz_m": xyz[-1].tolist()}
            sheet = Image.new("RGB", (256 * len(frames_at), 568), "white")
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 5), key + " | top: main camera; bottom: wrist", fill="black")
            for camera, y in (("image", 38), ("image2", 308)):
                found = set()
                with av.open(str(trial / (camera + ".mp4"))) as video:
                    for index, frame in enumerate(video.decode(video=0)):
                        if index in frames_at:
                            col = frames_at.index(index)
                            sheet.paste(frame.to_image(), (col * 256, y))
                            draw.text((col * 256 + 5, y - 14), f"step {index}", fill="black")
                            found.add(index)
                if found != set(frames_at):
                    raise RuntimeError("Missing selected video frame")
            sheet.save(attempt / "contact_sheet.png")
            result = {"case": key, "model": model, "initial_state": state, "instruction": instruction,
                      "goals": episode["goals"], "objects": objects, "input_hashes": inputs,
                      "contact_sheet": (attempt / "contact_sheet.png").relative_to(directory).as_posix(),
                      "annotation_status": "Unreviewed; quantitative motion only, not causal attribution"}
            atomic_json(attempt / "motion.json", result)
            journal.finish(key, result, [attempt / "contact_sheet.png", attempt / "motion.json"])
            outcomes.append(result)
        atomic_json(directory / "review_evidence.json", outcomes)
        print("PASS: eight selected dual-camera contact sheets and object-motion records saved. Visual interpretation is separate.", flush=True)


if __name__ == "__main__":
    main()
