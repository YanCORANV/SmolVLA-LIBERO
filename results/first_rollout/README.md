# First LIBERO rollout acceptance

Date: 2026-09-08. Suite: `libero_spatial`. Task ID: `0`. Seed: `0`.
Official result: **success=true, one episode**. Both evaluator and wrapper exited with 0.

The task language reconstructed from the installed LIBERO task map and default task
order is: "pick up the black bowl between the plate and the ramekin and place it on
the plate". The environment exposes this task description to the policy. This text
was checked against upstream code after the run, not captured as a per-step input log.

## Evidence

- [summary.csv](summary.csv): one row of outcome, provenance, timings and resource measurements.
- [summary.json](summary.json): measurement scopes, original video hash and upstream metrics.
- [keyframes.png](keyframes.png): frames 0, 15, 30, 46, 61 and 76, read left to right then top to bottom.
- Raw local artifacts: `outputs/first_rollout/`, excluded from Git.
- Local viewing copy: `outputs/first_rollout/videos/libero_spatial_0/eval_episode_0_playback_20fps.mp4`.

All 77 frames of the original H.264 video decoded without error. Visual inspection
shows the arm approach and grasp the black bowl, then move it onto the plate. The
visual evidence is consistent with LIBERO's success predicate; it does not establish
language causality. No extra rollout was performed to produce the summary.

## Video timing correction

The saved video uses 80 FPS and lasts 0.9625 s. The runtime configuration uses
`fps=20`, passed to the simulator as `control_freq`. The installed LeRobot LIBERO
wrapper instead advertises a constant `metadata['render_fps']=80`, and the evaluator
uses that metadata to encode video. Thus this recording plays four times faster than
its control-step cadence. The summary script makes a derived 20 FPS, 3.85 s viewing
copy, preserving all 77 frames. The original video and JSON are unchanged. Neither
video duration equals compute wall time.

## Interpretation and limitations

Evaluation took 114.39 s; the whole command took 362.84 s. Device-wide GPU samples
peaked at 2,534 MiB; maximum process RSS was 3,824.69 MiB. GPU measurements include
other GPU users and sample once per second. They may miss shorter spikes.

This verifies the integration pipeline on one task and initial condition. It does
not establish a 100% benchmark success rate, language robustness, or trainability
within 8 GB. The upstream CLI did not retain per-step actions or state/RNG snapshots;
whole-trajectory action finiteness and paired-state identity were not independently
audited. Add that instrumentation before controlled language comparisons. No
fine-tuning has been performed. No scientific claims are inferred from six frames.

The environment package list is recorded in `docs/environment-packages.txt`. Model
revision is fixed in the result tables. The referenced VLM/tokenizer revision and
simulator asset hashes need explicit capture for the subsequent benchmark. This run
preceded an initial Git commit and a clean-machine reproducibility check.
