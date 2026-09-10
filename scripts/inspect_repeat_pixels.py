"""Localize the first diagnostic action/input difference without inference.

Only two recorded entries and existing numerical traces are decoded. Outputs are
small and retained separately from the immutable run. No GPU imports are needed.
"""

import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.diagnostic_records import decode_record
from smolvla_grounding.libero_pairing import array_hashes
from smolvla_grounding.protocol import Checkpoints, atomic_json, file_sha256, fingerprint, run_lock


def numpy_tensor(metadata, arrays, key):
    spec = metadata["input"][key]
    if spec["dtype"] not in {"float32", "int64", "bool"}:
        raise ValueError(f"Unsupported diagnostic input type: {spec['dtype']}")
    raw = arrays["input/" + key]
    description = json.dumps(spec, sort_keys=True).encode()
    digest = hashlib.sha256(description + b"\0" + raw.tobytes()).hexdigest()
    if digest != metadata["trace"]["input"][key]:
        raise RuntimeError("Archived tensor bytes disagree with their inference-time hash")
    return raw.view(np.dtype(spec["dtype"])).reshape(spec["shape"])


def recover_pixels(tensor, camera, recorded_hash):
    if tensor.shape != (1, 3, 256, 256) or tensor.dtype != np.float32:
        raise ValueError("Expected the validated 256px float32 input")
    hwc = tensor[0].transpose(1, 2, 0)
    if not np.isfinite(hwc).all() or hwc.min() < 0 or hwc.max() > 1:
        raise ValueError("Image is outside the expected finite [0,1] scale")
    pixels = np.rint(hwc * 255).astype(np.uint8)
    if not np.array_equal(pixels.astype(np.float32) / 255, hwc):
        raise RuntimeError("Cannot recover exact original 8-bit image values")
    # Installed LiberoProcessorStep flips H and W. Undo it only for the raw hash
    # check; present the images in the policy's upright orientation.
    if array_hashes({camera: pixels[::-1, ::-1]})[camera] != recorded_hash:
        raise RuntimeError("Recovered image disagrees with the raw camera hash")
    return pixels


def pixel_stats(a, b):
    if a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("Expected same-shaped uint8 images")
    difference = b.astype(np.int16) - a.astype(np.int16)
    mask = np.any(difference != 0, axis=2)
    yy, xx = np.where(mask)
    return {"changed_pixels": len(xx), "total_pixels": int(mask.size),
            "changed_channels": int(np.count_nonzero(difference)),
            "max_abs_channel_delta_uint8": int(np.abs(difference).max()),
            "mean_abs_channel_delta_uint8": float(np.abs(difference).mean()),
            "bbox_xyxy_inclusive": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())] if len(xx) else None,
            "pixels": [{"x": int(x), "y": int(y), "original_rgb": a[y, x].tolist(),
                        "repeat_rgb": b[y, x].tolist()} for x, y in zip(xx, yy, strict=True)]}, mask


def draw_comparison(a, b, mask, stats, output):
    canvas = Image.new("RGB", (3 * 256, 2 * 290), "white")
    draw = ImageDraw.Draw(canvas)
    marked = np.zeros_like(a)
    marked[mask] = [255, 0, 0]
    box = stats["bbox_xyxy_inclusive"]
    if box:
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        x0, y0 = max(0, min(cx - 16, 224)), max(0, min(cy - 16, 224))
    else:
        x0, y0 = 112, 112
    crop = (x0, y0, x0 + 32, y0 + 32)
    for col, (pixels, title) in enumerate(((a, "Original"), (b, "Repeat"), (marked, "Changed pixels (red)"))):
        frame = Image.fromarray(pixels)
        canvas.paste(frame, (col * 256, 28))
        draw.text((col * 256 + 6, 7), title, fill="black")
        draw.rectangle((col * 256 + x0, 28 + y0, col * 256 + x0 + 32, 28 + y0 + 32), outline="red")
        enlarged = frame.crop(crop).resize((256, 256), Image.Resampling.NEAREST)
        canvas.paste(enlarged, (col * 256, 318))
        draw.text((col * 256 + 6, 296), "32 x 32 crop, nearest-neighbor 8x", fill="black")
    canvas.save(output)


def main():
    source = ROOT / "outputs/repeat_diagnostic"
    destination = ROOT / "outputs/repeat_pixel_review"
    report = json.loads((source / "diagnostic.json").read_text())
    identity = json.loads((source / "identity.json").read_text())
    if not report["execution_complete"]:
        raise RuntimeError("Diagnostic execution must be complete")
    step = report["trace_comparison"]["first_different_step"]["normalized_action"]
    if step is None:
        raise RuntimeError("This diagnostic contains no differing action to inspect")
    with run_lock(destination):
        print("[1/3] Verify diagnostic markers and decode the first differing input", flush=True)
        run_fingerprint = fingerprint(identity)
        markers = []
        decoded, trajectories = [], []
        for trial in report["trials"]:
            name = trial["instruction_id"]
            marker = json.loads((source / name / "complete.json").read_text())
            if marker["fingerprint"] != run_fingerprint or marker["result"] != trial:
                raise RuntimeError("Diagnostic completion record mismatch")
            for filename, expected in marker["artifacts"].items():
                if file_sha256(source / filename) != expected:
                    raise RuntimeError(f"Diagnostic artifact modified: {filename}")
            attempt = source / trial["attempt"]
            with zipfile.ZipFile(attempt / "inference.zip") as archive:
                decoded.append(decode_record(archive.read(f"step_{step:04d}.npz")))
            with np.load(attempt / "trajectory.npz", allow_pickle=False) as saved:
                trajectories.append({k: saved[k].copy() for k in ("action", "sim_state")})
            markers.append(marker)
        checkpoints = Checkpoints(destination, {"source_identity": run_fingerprint,
                    "diagnostic_sha256": file_sha256(source / "diagnostic.json"),
                    "script_sha256": file_sha256(Path(__file__)), "step": step})
        comparisons = []
        print("[2/3] Recover original pixel values and compare both cameras", flush=True)
        for camera in ("image", "image2"):
            completed = checkpoints.completed(camera)
            if completed is None:
                output = checkpoints.new_attempt(camera)
                images = [recover_pixels(numpy_tensor(meta, arrays, "observation.images." + camera),
                                         camera, meta["trace"]["raw_pixels"][camera]) for meta, arrays in decoded]
                stats, mask = pixel_stats(*images)
                for name, frame in zip(("original", "repeat"), images, strict=True):
                    Image.fromarray(frame).save(output / (name + ".png"))
                draw_comparison(*images, mask, stats, output / "comparison.png")
                atomic_json(output / "pixels.json", stats)
                with (output / "pixels.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["policy_x", "policy_y", "original_r", "original_g", "original_b", "repeat_r", "repeat_g", "repeat_b"])
                    for pixel in stats["pixels"]:
                        writer.writerow([pixel["x"], pixel["y"], *pixel["original_rgb"], *pixel["repeat_rgb"]])
                completed = {"camera": camera, "stats": stats, "raw_hash_verified": True,
                             "comparison": (output / "comparison.png").relative_to(destination).as_posix()}
                checkpoints.finish(camera, completed, [output / n for n in
                                   ("original.png", "repeat.png", "comparison.png", "pixels.json", "pixels.csv")])
            comparisons.append(completed)
            print(f"  {camera}: {completed['stats']['changed_pixels']} changed pixels; "
                  f"maximum 8-bit channel difference {completed['stats']['max_abs_channel_delta_uint8']}", flush=True)
        print("[3/3] Align observations with action/state indices and save evidence", flush=True)
        equality = {key: np.array_equal(numpy_tensor(*decoded[0], key), numpy_tensor(*decoded[1], key))
                    for key in decoded[0][0]["input"]}
        a, b = trajectories
        review = {"diagnostic_identity_fingerprint": run_fingerprint, "policy_step": step,
                  "observation_after_action": step - 1,
                  "actions_equal_before_this_step": bool(np.array_equal(a["action"][:step - 1], b["action"][:step - 1])),
                  "saved_sim_states_equal_through_observation": bool(np.array_equal(a["sim_state"][:step], b["sim_state"][:step])),
                  "input_tensor_equality": {k: bool(v) for k, v in equality.items()},
                  "cameras": comparisons, "source_replay": report["replay"],
                  "interpretation": "A small primary-camera difference is already present before action selection. Its exact rendering/hidden-state cause remains unproven.",
                  "limitations": ["Saved simulator arrays do not cover every derived renderer/controller state.",
                                  "Replays establish repeatability for these recorded inputs only.",
                                  "This checks an observed natural perturbation, not adversarial or population robustness."]}
        atomic_json(destination / "pixel_review.json", review)
        print("PASS: exact source pixel hashes verified; camera localization and state alignment saved.", flush=True)


if __name__ == "__main__":
    main()
