# Resource-aware evaluation concurrency

Update: the 1000-update evaluation completed with actual process overlap. Its
scheduling stage took about 2 h 4 min; the prior serial command took 1 h 47 min.
No speedup was demonstrated, and the comparison uses different checkpoints/timing
scopes. See [measured results](../results/longer_training_development/README.md).
For the next validation, enter 1 at the concurrency prompt; historical source
and experiment identities are preserved.

The 1000-update evaluation entry point now asks for the maximum number of concurrent
evaluation processes: **2 by default**, or 1 for serial execution. This is the
parallelism requested by the project owner; it is an interactive choice, not a
command-line mode. The choice is frozen in `experiment.json` and retained on resume.

```bash
bash scripts/run_training_budget_evaluation.sh
```

The first prompt selects the run name; the second selects the maximum process count.
Existing completed evaluations and historical entry points are unchanged.

## Why available VRAM is not sufficient evidence

Measured on this installation:

| Resource | Observation |
| --- | --- |
| GPU VRAM | 8188 MiB total |
| Earlier 36-episode run, whole GPU | Median utilization 34%; maximum memory 2963 MiB, including rendering/runtime |
| Per-episode PyTorch reserved | Approximately 1.43–1.46 GiB in the compared evaluation |
| Whole-command maximum process RSS | 3.94 GiB in the later 24-episode run; 4.80 GiB in the earlier 36-episode run |
| Current WSL RAM | 7790 MiB total; 6916 MiB available when inspected idle |
| Earlier 24-episode wall time | 1 h 47 min, including startup and verification |

GPU statistics are sampled across the entire older run, including startup, and RSS
is a peak, not the process's steady memory footprint. The data indicate possible
GPU headroom but do not demonstrate a speedup. Two concurrent processes may exceed
WSL RAM even if VRAM fits. CPU contention can also reduce multiprocessing efficiency;
see [PyTorch's multiprocessing guidance](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html).
This change preserves existing CPU thread and GPU inference settings so it does not
silently change the experiment's precision, image resolution or action horizon.

## Scheduling and recorded evidence

The original-language and augmented-language models run in separate processes and
write separate journals. All shared initial-state and RNG records are verified and
committed before workers launch; the workers read those shared records. Their seeds,
initial observation checks, checkpoint hashes and normalization checks remain intact.
This preserves protocol checks; it is not a claim of bitwise-identical trajectories
under concurrent execution. Prior rendering variability remains relevant.

The second process starts only after the first rewrites its successful model-load
report and system-wide available memory is at least **5.25 GiB RAM and 3.25 GiB VRAM**.
These thresholds allow margin above prior measured peaks, including cold loading.
They are admission estimates, not guarantees against OOM or slowdown. With the
current 7.6 GiB WSL allocation, the second worker may wait until the first finishes,
leaving the run serial. No WSL memory setting is changed automatically.

The terminal reports active/waiting workers, GPU utilization and free RAM/VRAM every
30 seconds, with a short per-worker log tail. `scheduling/<execution>/` contains
individual worker console logs, sampled `resources.jsonl`, timestamps, exit codes
and `summary.json`. `ever_overlapped` explicitly distinguishes requested concurrency
from concurrency actually obtained. Different checkpoints may have different rollout
timings, so comparison against historical runtime is descriptive, not a controlled
performance benchmark. Real GPU concurrency has now been observed; speedup has not
been established.

On worker failure or user interruption, the scheduler stops only worker process
groups it launched and preserves committed episode checkpoints. Rerunning the same
command/name asks to resume; an interrupted worker may also require confirming stale
lock recovery. The scheduler does not silently accept partial episodes or count
them as failures. Short CPU-only process tests verified overlap, successful exit,
failure cleanup and resource-admission behavior in WSL; they loaded no model/simulator.

If RAM consistently prevents overlap, reassess measured steady and peak RAM before
changing WSL's allocation or implementing shared-policy batched inference. Batched
inference would require separate checks of per-environment randomness and numerical
behavior; it is not included in this change.
