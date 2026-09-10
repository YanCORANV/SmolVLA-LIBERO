"""Validate the installed Linux environment without downloading policy weights."""

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone


def main():
    output = Path(__file__).resolve().parents[1] / "outputs" / "setup"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report = {"time_utc": stamp, "platform": platform.system(),
              "python": platform.python_version(), "checks": {}, "passed": False}
    try:
        print("[1/4] Validate Linux and Python", flush=True)
        if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
            raise RuntimeError("Expected Linux and Python 3.12 in the project environment")
        packages = ("lerobot", "hf-libero", "torch", "torchvision", "torchcodec",
                    "mujoco", "transformers", "robosuite", "av")
        report["packages"] = {p: importlib.metadata.version(p) for p in packages}
        print(json.dumps(report["packages"], indent=2), flush=True)
        report["checks"]["runtime"] = True

        print("[2/4] Execute a CUDA matrix multiplication", flush=True)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-enabled PyTorch cannot access the GPU")
        x = torch.eye(32, device="cuda")
        if not torch.equal(x @ x, x):
            raise RuntimeError("CUDA matrix multiplication returned an unexpected value")
        torch.cuda.synchronize()
        report["gpu"] = {"name": torch.cuda.get_device_name(0),
                         "torch_cuda": torch.version.cuda,
                         "bf16_supported": torch.cuda.is_bf16_supported(),
                         "total_bytes": torch.cuda.get_device_properties(0).total_memory}
        report["checks"]["cuda_tensor_operation"] = True

        print("[3/4] Import SmolVLA, LIBERO and video dependencies", flush=True)
        for module in ("lerobot.policies.smolvla.modeling_smolvla",
                       "libero.libero.benchmark", "av", "torchcodec"):
            importlib.import_module(module)
            print("Imported " + module, flush=True)
        report["checks"]["imports"] = True

        print("[4/4] Render one MuJoCo frame (not a LIBERO rollout)", flush=True)
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><light pos="0 0 3"/>'
            '<geom type="sphere" size="0.1" rgba="1 0 0 1"/>'
            '</worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=64, width=64) as renderer:
            renderer.update_scene(data)
            frame = renderer.render()
        if frame.shape != (64, 64, 3) or int(frame.max()) == int(frame.min()):
            raise RuntimeError("Offscreen renderer produced an invalid or constant frame")
        report["checks"]["mujoco_render"] = True
        report["render_backend"] = os.environ["MUJOCO_GL"]
        report["passed"] = True
        print("PASS: environment only; pretrained policy rollout is still pending.", flush=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        path = output / ("environment_" + stamp + ".json")
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Saved " + str(path.relative_to(output.parent.parent)), flush=True)


if __name__ == "__main__":
    main()
