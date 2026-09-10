"""Download the pinned official policy with Hugging Face's resumable downloader."""

from datetime import datetime, timezone
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/checkpoint.json").read_text(encoding="utf-8"))
    target = root / config["local_dir"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report = {"time_utc": stamp, "repo_id": config["repo_id"],
              "revision": config["revision"], "local_dir": config["local_dir"],
              "download_verified": False, "policy_loaded": False}
    try:
        print("[1/2] Download official policy and processors", flush=True)
        print(config["repo_id"] + " @ " + config["revision"], flush=True)
        print("Interrupted downloads can reuse the Hub cache when this script is rerun.", flush=True)
        snapshot_download(repo_id=config["repo_id"], revision=config["revision"], local_dir=target)
        print("[2/2] Check required snapshot files", flush=True)
        required = (
            "config.json", "model.safetensors", "policy_preprocessor.json",
            "policy_postprocessor.json",
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
            "policy_postprocessor_step_1_unnormalizer_processor.safetensors",
        )
        report["file_sizes_bytes"] = {}
        for name in required:
            path = target / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError("Required checkpoint file is missing or empty: " + name)
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            report["file_sizes_bytes"][name] = path.stat().st_size
        report["download_verified"] = True
        print("PASS: checkpoint files downloaded; model loading and rollout are still pending.", flush=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        output = root / "outputs/setup" / ("checkpoint_download_" + stamp + ".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Saved " + output.relative_to(root).as_posix(), flush=True)


if __name__ == "__main__":
    main()
