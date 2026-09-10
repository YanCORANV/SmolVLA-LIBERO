"""Load the official checkpoint and measure one synthetic-input prediction.

This validates model execution only. It does not run or score a LIBERO episode.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time


def main():
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report = {"time_utc": stamp, "passed": False, "input_source": "synthetic",
              "libero_rollout": False, "seed": 0}
    torch = None
    started = time.perf_counter()
    try:
        print("[1/4] Read the fixed checkpoint configuration", flush=True)
        import torch
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.utils.random_utils import set_seed

        checkpoint = json.loads((root / "configs/checkpoint.json").read_text(encoding="utf-8"))
        path = root / checkpoint["local_dir"]
        report.update({"repo_id": checkpoint["repo_id"], "revision": checkpoint["revision"]})
        report["config_sha256"] = hashlib.sha256((path / "config.json").read_bytes()).hexdigest()
        config = SmolVLAConfig.from_pretrained(path)
        config.device = "cuda"
        config.push_to_hub = False
        if config.use_amp:
            raise RuntimeError("Expected the downloaded checkpoint's use_amp=false setting")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        set_seed(report["seed"])
        torch.cuda.reset_peak_memory_stats()
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["settings"] = {k: getattr(config, k) for k in (
            "use_amp", "load_vlm_weights", "vlm_model_name", "chunk_size",
            "n_action_steps", "num_steps", "resize_imgs_with_padding")}

        print("[2/4] Load policy with strict checkpoint-key matching", flush=True)
        print("The official loader may download SmolVLM backbone/processor files on first use.", flush=True)
        load_start = time.perf_counter()
        policy = SmolVLAPolicy.from_pretrained(path, config=config, strict=True)
        policy.eval()
        torch.cuda.synchronize()
        report["load_seconds"] = time.perf_counter() - load_start
        report["load_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["load_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["strict_checkpoint_load"] = True
        dtypes = Counter()
        for parameter in policy.parameters():
            dtypes[str(parameter.dtype)] += parameter.numel()
        report["parameter_counts_by_dtype"] = dict(dtypes)
        print("Loaded parameter counts: " + json.dumps(dict(dtypes)), flush=True)

        print("[3/4] Load checkpoint normalization and tokenizer processors", flush=True)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        # Batch dimension is explicit. Images use the official [0,1] float input scale.
        observation = {name: torch.zeros((1, *feature.shape), dtype=torch.float32)
                       for name, feature in config.input_features.items()}
        observation["task"] = ["pick up the black bowl and place it on the plate"]
        report["synthetic_instruction"] = observation["task"][0]
        report["input_shapes"] = {k: list(v.shape) for k, v in observation.items()
                                  if isinstance(v, torch.Tensor)}
        policy.reset()

        print("[4/4] Predict one action and record memory (no simulator)", flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        predict_start = time.perf_counter()
        with torch.inference_mode():
            batch = preprocessor(observation)
            normalized_action = policy.select_action(batch)
            action = postprocessor(normalized_action)
        torch.cuda.synchronize()
        report["first_prediction_seconds"] = time.perf_counter() - predict_start
        expected_shape = (1, config.action_feature.shape[0])
        if tuple(action.shape) != expected_shape or not torch.isfinite(action).all().item():
            raise RuntimeError("Action must have shape " + str(expected_shape) + " and finite values")
        report["action_shape"] = list(action.shape)
        report["action"] = action.detach().cpu().tolist()
        report["prediction_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["prediction_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        free, total = torch.cuda.mem_get_info()
        report["device_free_bytes_after_prediction"] = free
        report["device_total_bytes"] = total
        report["memory_scope"] = "PyTorch peaks only; no LIBERO renderer, not full-device peak usage"
        report["passed"] = True
        peak = max(report["load_peak_reserved_bytes"], report["prediction_peak_reserved_bytes"])
        print(f"PASS: finite action {list(action.shape)}; peak PyTorch reserved {peak / 2**30:.2f} GiB.", flush=True)
        print("LIBERO scene execution and success rate remain unvalidated.", flush=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        output = root / "outputs/setup" / ("policy_validation_" + stamp + ".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Saved " + output.relative_to(root).as_posix(), flush=True)


if __name__ == "__main__":
    main()
