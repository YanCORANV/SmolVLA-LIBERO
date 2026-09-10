"""Fresh-process reload of a profiled policy and official optimizer state."""

import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.protocol import atomic_json, file_sha256


def main():
    directory = Path(os.environ["SMOLVLA_TRAINING_PROFILE_DIR"])
    report = {"passed": False}
    started = time.perf_counter()
    try:
        print("[reload 1/3] Verify saved artifacts in a fresh process", flush=True)
        previous = json.loads((directory / "step_report.json").read_text())
        if not previous["passed"]:
            raise RuntimeError("Optimizer-step acceptance did not pass")
        for name, expected in previous["artifacts"].items():
            if file_sha256(directory / name) != expected:
                raise RuntimeError("Profile checkpoint artifact changed: " + name)
        import numpy as np
        import torch
        from huggingface_hub import try_to_load_from_cache
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.common.train_utils import load_training_state
        from lerobot.utils.collate import lerobot_collate_fn
        from smolvla_grounding.training_policy import FP32ExpertSmolVLAPolicy, parameter_evidence

        path = directory / "checkpoint/pretrained_model"
        config = SmolVLAConfig.from_pretrained(path)
        cached = try_to_load_from_cache(previous["backbone"]["repo_id"], "config.json", revision=previous["backbone"]["revision"])
        if not isinstance(cached, str):
            raise RuntimeError("Pinned backbone cache missing")
        config.vlm_model_name, config.device = str(Path(cached).parent), "cuda"
        print("[reload 2/3] Strictly reload policy and compare every parameter", flush=True)
        policy = FP32ExpertSmolVLAPolicy.from_pretrained(path, config=config, strict=True).eval()
        actual, counts = parameter_evidence(policy)
        expected = json.loads((directory / "parameters_after.json").read_text())
        if actual != expected:
            raise RuntimeError("Reload changed parameter values, dtype or frozen status")
        optimizer = config.get_optimizer_preset().build([p for p in policy.parameters() if p.requires_grad])
        step, optimizer, _ = load_training_state(directory / "checkpoint", optimizer, None)
        if step != 1 or not optimizer.state:
            raise RuntimeError("Official optimizer state/step did not restore")
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                    raise RuntimeError("Nonfinite reloaded optimizer state")
        report["strict_reload"] = True
        report["all_parameter_bytes_equal"] = True
        report["parameter_counts"] = counts
        report["optimizer_step"] = step
        report["optimizer_state_entries"] = len(optimizer.state)
        del optimizer
        torch.cuda.empty_cache()
        print("[reload 3/3] Load saved processors and predict from the recorded real observation", flush=True)
        pre, post = make_pre_post_processors(config, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": "cuda"}, "tokenizer_processor": {"tokenizer_name": config.vlm_model_name}})
        with np.load(directory / "sample.npz", allow_pickle=False) as archive:
            observation = {key: torch.from_numpy(archive[key].copy()) for key in archive.files
                           if key.startswith("observation.")}
        observation["task"] = previous["dataset"]["instruction"]
        policy.reset()
        with torch.inference_mode():
            action = post(policy.select_action(pre(lerobot_collate_fn([observation]))))
        if action.shape != (1, 7) or not torch.isfinite(action).all():
            raise RuntimeError("Reloaded policy failed finite-action check")
        report["action"] = action.cpu().tolist()
        report["passed"] = True
        report["scope"] = "One real optimizer update and fresh-process policy/optimizer reload; no rollout or robustness gain measured"
        print("PASS: real training step and fresh-process reload completed.", flush=True)
        print(f"Update: {previous['update']['wall_seconds']:.2f} s; peak reserved {previous['update']['torch_peak_reserved_bytes']/2**30:.2f} GiB.", flush=True)
        print("This is first-step profiling, not steady-state training speed or a fine-tuning result.", flush=True)
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        atomic_json(directory / "reload_report.json", report)
        print("Saved " + (directory / "reload_report.json").relative_to(ROOT).as_posix(), flush=True)


if __name__ == "__main__":
    main()
