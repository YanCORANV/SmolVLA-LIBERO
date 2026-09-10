"""Resource-aware serial/two-process scheduling for independent policy workers.

Admission thresholds are estimates from this project's measured peaks, not OOM
guarantees. All reference records must exist before concurrent workers start.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .protocol import atomic_json

GIB = 1024**3
# Prior maximum evaluator RSS was 4.80 GiB and whole-device use about 2.90 GiB.
# Reserve headroom before starting a second process, including cold model loading.
ADDITIONAL_RAM_BYTES = int(5.25 * GIB)
ADDITIONAL_GPU_BYTES = int(3.25 * GIB)


def can_add_worker(snapshot, active_count, maximum, all_active_ready):
    if active_count >= maximum:
        return False
    if active_count == 0:
        return True  # The accepted one-process configuration remains available.
    return (all_active_ready and snapshot["available_ram_bytes"] >= ADDITIONAL_RAM_BYTES
            and snapshot["gpu_free_bytes"] >= ADDITIONAL_GPU_BYTES)


def resources():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=10)
    devices = result.stdout.strip().splitlines()
    if len(devices) != 1:
        raise RuntimeError("This resource scheduler expects the project's single GPU")
    memory, utilization = [x.strip() for x in devices[0].split(",")]
    return {"available_ram_bytes": values["MemAvailable"], "swap_used_bytes": values["SwapTotal"] - values["SwapFree"],
            "gpu_free_bytes": int(memory) * 1024**2, "gpu_utilization_percent": float(utilization)}


def run_workers(root, directory, models, maximum):
    if maximum not in (1, 2):
        raise ValueError("Use one or two evaluation processes")
    root, directory = Path(root), Path(directory)
    pending, active, logs = list(models), {}, {}
    attempt = directory / "scheduling" / str(time.time_ns())
    attempt.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {"passed": False, "maximum_workers": maximum, "ever_overlapped": False,
              "admission_ram_bytes": ADDITIONAL_RAM_BYTES, "admission_gpu_bytes": ADDITIONAL_GPU_BYTES,
              "workers": {}, "limits": "Measured-peak admission estimate; no speedup or OOM guarantee"}
    next_notice = 0
    try:
        with (attempt / "resources.jsonl").open("w", encoding="utf-8") as monitor:
            while pending or active:
                for model, process in list(active.items()):
                    status = process.poll()
                    if status is not None:
                        report["workers"][model].update(exit_code=status, end_seconds=time.perf_counter() - start)
                        del active[model]
                        logs[model].close()
                        print(f"  Worker {model} exited with status {status}.", flush=True)
                        if status:
                            raise RuntimeError(f"{model} failed; see {report['workers'][model]['console']}. Completed episodes are preserved.")
                snapshot = resources()
                elapsed = time.perf_counter() - start
                monitor.write(json.dumps({"elapsed_seconds": elapsed, "active": list(active), **snapshot}) + "\n")
                monitor.flush()
                # A previous model_load.json does not prove the new process has
                # loaded its policy: require a file rewritten after this launch.
                ready = all((directory / m / "model_load.json").exists()
                            and (directory / m / "model_load.json").stat().st_mtime_ns >= report["workers"][m]["launch_time_ns"]
                            for m in active)
                if pending and can_add_worker(snapshot, len(active), maximum, ready):
                    model = pending.pop(0)
                    log_path = attempt / (model + ".log")
                    logs[model] = log_path.open("w", encoding="utf-8")
                    launch_time = time.time_ns()
                    process = subprocess.Popen([sys.executable, "-u", str(root / "scripts/evaluate_adaptation_model.py")],
                        cwd=root, stdout=logs[model], stderr=subprocess.STDOUT,
                        env=dict(os.environ, SMOLVLA_ADAPTATION_EVAL_DIR=str(directory), SMOLVLA_ADAPTATION_EVAL_MODEL=model),
                        start_new_session=True)
                    active[model] = process
                    report["workers"][model] = {"pid": process.pid, "launch_time_ns": launch_time,
                        "start_seconds": time.perf_counter() - start, "console": log_path.relative_to(root).as_posix()}
                    report["ever_overlapped"] |= len(active) > 1
                    print("  Start " + model + "; log: " + log_path.relative_to(root).as_posix(), flush=True)
                    atomic_json(attempt / "summary.json", report)
                if elapsed >= next_notice:
                    print(f"  Elapsed {elapsed/60:.1f} min; active={list(active)}; waiting={pending}; "
                          f"RAM available={snapshot['available_ram_bytes']/GIB:.2f} GiB; "
                          f"GPU free={snapshot['gpu_free_bytes']/GIB:.2f} GiB; utilization={snapshot['gpu_utilization_percent']:.0f}%.", flush=True)
                    for model in active:
                        # Read a bounded log tail for visible per-worker progress.
                        with (root / report["workers"][model]["console"]).open("rb") as stream:
                            stream.seek(0, 2)
                            stream.seek(max(0, stream.tell() - 1500))
                            tail = stream.read().decode("utf-8", errors="replace").splitlines()
                        if tail:
                            print("    " + model + ": " + tail[-1][-250:], flush=True)
                    if pending and active and maximum == 2:
                        print("    Second worker waits for policy load and >=5.25 GiB RAM / >=3.25 GiB GPU free; otherwise this run stays serial.", flush=True)
                    next_notice = elapsed + 30
                if pending or active:
                    time.sleep(3)
        report["passed"] = True
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        # Stop only processes launched here; their committed episode checkpoints
        # survive. Signal groups also include any video encoder child processes.
        import signal
        for process in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for model, process in active.items():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            report["workers"][model].update(exit_code=process.returncode, end_seconds=time.perf_counter()-start)
        for stream in logs.values():
            if not stream.closed:
                stream.close()
        report["elapsed_seconds"] = time.perf_counter() - start
        atomic_json(attempt / "summary.json", report)
        print("Scheduling report: " + (attempt / "summary.json").relative_to(root).as_posix(), flush=True)
