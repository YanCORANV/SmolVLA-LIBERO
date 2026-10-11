# Reproduction notes

This is the staged workflow that ran in the project's Linux/WSL2 environment. It is **not** a verified one-command clean-clone reproduction: official datasets/model weights, local source checkpoints and prior-stage reports are not committed, and several frozen configs refer to those local artifacts. Raw GPU training and rollouts can take many hours. The [setup guide](setup.md) covers the environment.

## Orientation

The current scientific result is in [results.md](results.md), with small machine-readable copies under [results/](../results/README.md). Do not interpret historical development notes as final evidence. No additional training or test rollout is needed to view or regenerate the portfolio figures.

From the repository root in Linux/WSL2, the early integration path was:

```bash
bash scripts/setup_environment.sh
source .venv/bin/activate
python scripts/download_checkpoint.py
python scripts/check_policy.py
bash scripts/run_first_rollout.sh
python scripts/summarize_first_rollout.py
```

The official checkpoint is `HuggingFaceVLA/smolvla_libero`, pinned in this project to revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. See [requirements](../requirements.txt) and [recorded environment versions](environment-packages.txt). The direct dependency pins are not a complete portable lockfile.

## Completed experiment entry points

These are the *actual scripts* used by the staged study. Each requires successful artifacts from preceding stages and may refuse to run when frozen source/config hashes differ. They are shown for inspection and controlled reproduction, not as a recommendation to rerun consumed states.

| Stage | Entrypoint | Required context |
| --- | --- | --- |
| Pairing and pilot | `python scripts/check_language_pairing.py`, `bash scripts/run_language_pilot.sh` | Official checkpoint, simulator and language catalogs |
| Prepare task data | `python scripts/prepare_task0_data.py` | Official exact-task LIBERO demonstrations |
| Original/augmented training history | `bash scripts/run_adaptation.sh`, `bash scripts/run_training_budget.sh` | Prepared data and prior study reports; see [historical training guide](paired_finetuning.md) |
| First confirmatory held-out evaluation | `python scripts/run_held_out_evaluation.py` | Frozen [first matrix](../configs/held_out_evaluation/confirmatory_matrix.json) and all three checkpoints |
| Anchored sampling | `python scripts/run_anchored_training.py` | Frozen [anchored config](../configs/anchored_training.json) |
| Anchored development comparison | `python scripts/run_development_evaluation.py` | States 23–32, anchored/canonical/augmented checkpoints |
| Action/trajectory diagnosis | `python scripts/diagnose_language_divergence.py` | Existing held-out and development trajectories; no new rollouts |
| Paired BC and consistency training | `python scripts/run_paired_training.py` | Frozen [paired config](../configs/paired_training.json), Task-0 ready data and canonical step-1000 checkpoint |
| Paired development selection | `python scripts/run_paired_development_evaluation.py` | Both committed paired checkpoints, States 23–32 |
| Amended final v4 evaluation | `python scripts/run_final_confirmatory_evaluation.py` | Frozen [v4 matrix](../configs/final_confirmatory/final_matrix_v4.json), model manifest, final prompts and active 50-state asset |

The final entrypoint has completed **153/153** valid rollouts. Its `complete.json` journal skips committed trials and retains incomplete attempts; running it again does not create a new independent test. States 33–49 and their final prompts are now consumed for this project. The initially configured 50–52 are out of range in the active Task-0 initialization asset. [Amendment audit](../configs/final_confirmatory/freeze_final_protocol_v4.json).

## Rebuild public figures without simulation

The curated input copies are checked in. This command reads only the existing [diagnostic CSV](../results/language_divergence/pair_level_divergence_measurements.csv), [development summaries](../results/development_selection/) and [final summary](../results/final_confirmatory_v4/summary.json):

```bash
python scripts/make_portfolio_figures.py
```

It writes three PNGs under `assets/figures/`. It does not load SmolVLA, reset a LIBERO state, or run inference.

## Separate five-task development pilot

This is **Study B**, using the original untuned SmolVLA policy. It does not test the Task-0 Paired BC checkpoint. The frozen [matrix](../configs/multitask_evaluation/language_pilot_matrix.json) and [manifest](../configs/multitask_evaluation/language_pilot_manifest.json) fix the official checkpoint revision, its artifact hashes, five prompt catalogs and hashes, five task/state assets and hashes, 75 identities, worker count, horizons, seeds, and success semantics. The eligible tasks were selected with a 3/5 canonical competence gate on States 0–4; this selection biases the pilot toward tasks the untuned policy could already perform. The pilot uses States 5–9 only; States 30–49 remain reserved.

The actual project runtime is WSL2 Ubuntu-20.04, Python **3.12.14**, torch **2.10.0+cu128**, LeRobot **0.6.1**, robosuite **1.4.0**, MuJoCo **3.3.2**, NumPy **2.2.6**, Matplotlib **3.11.1**, and OpenCV **5.0.0.93**. LIBERO is present under the project venv but is not registered as a versioned Python distribution; use the hashed BDDL and pruned-init assets in the manifest. The execution machine used an NVIDIA RTX 4070 Laptop GPU (8188 MiB reported VRAM, driver 592.82). The direct dependencies are in [requirements.txt](../requirements.txt); [environment-packages.txt](environment-packages.txt) records more installed versions, but is not a portable lockfile.

With the official checkpoint, backbone, LIBERO assets, dependencies and source hashes in place, the fixed entrypoints from the repository root are:

~~~bash
.venv/bin/python scripts/run_multitask_language_pilot.py
.venv/bin/python scripts/analyze_multitask_language_pilot.py
.venv/bin/python scripts/export_multitask_failure_cases.py
~~~

The runner prints progress and skips each hash-validated complete.json cell. It creates a fresh attempt directory for an incomplete cell and uses a run lock to prevent two writers. It stops on pairing or frozen-hash mismatch. The existing 75 trials are consumed development data; the commands document controlled reproduction, not a request to run more trials. Raw commits, trajectories, images, inference traces and logs remain under outputs/multitask_evaluation/runs/multitask_language_sensitivity_pilot/ and are Git-ignored.

For **independent published-data regeneration without a GPU, checkpoint, simulator or raw videos**, use a Python environment with NumPy and Matplotlib and run:

~~~bash
python scripts/regenerate_multitask_language_pilot.py
~~~

That command reads the committed [75-row source table](../results/multitask_language_pilot/per_rollout.csv), [50-row pair table](../results/multitask_language_pilot/pair_level.csv), [summary](../results/multitask_language_pilot/summary.json), [case index](../results/multitask_language_pilot/case_index.csv), frozen matrix/catalogs, and compact EEF example data. It recomputes success counts, goal_final counts, paired transitions and divergence aggregates, checks every case metric and sheet hash, renders the four quantitative plots to a temporary directory, and compares their hashes to the [integrity audit](../results/multitask_language_pilot/integrity.json). Matching figure bytes were verified in the project venv; other font/rendering versions can alter PNG bytes without changing numeric results. The published-data check cannot prove the underlying local simulator outcomes; the raw analyzer verifies complete.json and every hashed source artifact when that local evidence is present.

The 12 case sheets are derived from the two existing image.mp4 files per selected pair and can be regenerated by the exporter only where the ignored raw videos exist. A clean-clone model/simulator rollout reproduction on another machine has **not** been executed.
## Audit, resume and local artifacts

- `outputs/`, `data/`, `checkpoints/`, `.venv/`, raw MP4s and model weights are Git-ignored. The public repository contains code, configs, protocol hashes, selected summary/CSV copies and figures.
- Training checkpoints are committed through the existing benchmark journal. Evaluations count only attempts with a valid `complete.json`; interrupted attempts remain available for forensic review.
- The [frozen model manifest](../configs/final_confirmatory/frozen_model_manifest_v4.json) identifies checkpoints; the [v4 freeze](../configs/final_confirmatory/freeze_final_protocol_v4.json) records source/config/model/prompt/state-asset hashes and the corrected matrix.
- The final uncertainty calculation is a 10,000-replicate paired bootstrap over **17 state IDs**, keeping the three models and three prompt outcomes together for each sampled state.
- CPU record/protocol checks can be run with `python -m unittest discover -s tests -v`; real SmolVLA checks need the configured project environment. The full suite was last run in the project WSL `.venv` before the v4 freeze.

A new independent study needs its own source artifacts, language catalog, state partition and preregistered protocol. The existing test prompts and State 33–49 outcomes must not be reused to select another method.
