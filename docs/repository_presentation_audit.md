# Repository presentation audit

This pass changes the public explanation and derived presentation assets. It does **not** change model weights, training runs, frozen prompts, rollout outcomes, statistical rules, or source evaluation protocols.

| Finding | Evidence | Presentation action |
| --- | --- | --- |
| Root README described a three-state v0.1 prototype as current | Previous README's latest table was States 0–2 | Replaced the landing page with the completed pipeline, explicit negative final result and links to detailed evidence |
| Old results index called the 1,000-update *development* round the latest result | Previous `results/README.md` | Added first confirmatory, development, diagnostic and final v4 curated data/index entries |
| Historical handoff still names planned States 33–52 | `docs/agent_handoff_zh.md` | Marked it historical; current final v4 is 33–49 because the active Task-0 asset has only indices 0–49 |
| Some older stage documents still contain pending milestones | `docs/language_pilot.md`, `docs/paired_finetuning.md`, `docs/held_out_evaluation.md`, `docs/publish_github_zh.md` and peers | Preserved them as contemporaneous records; the documentation index now routes readers to current `experiments.md`, `method.md` and `results.md` |
| Small frozen result summaries lived only in ignored `outputs/` | Diagnostic, development, first confirmatory and final local outputs | Copied byte-identical JSON/CSV into `results/` so public figures can be rebuilt without a rollout |
| Useful diagnostic plots existed only in ignored outputs and had crowded labels | `outputs/language_divergence_diagnostics/plots/` | Generated three focused GitHub figures directly from curated CSV/JSON using one short script |
| Completed rollout MP4s remain local | `outputs/final_confirmatory/runs/.../image.mp4` and `image2.mp4` | Preserved raw video files; no additional rollout or video edit was needed for this pass |
| Frozen final entrypoint docstring still says 180 although v4 executes 153 | `scripts/run_final_confirmatory_evaluation.py` versus frozen v4 matrix | Leave frozen runtime bytes unchanged; current README/protocol/results state 153 explicitly |
| Experimental provenance was mostly untracked relative to the initial public commit | Git status at start of this pass | Keep historical experiment files intact and group repository updates by pipeline/protocol, documentation and derived visuals rather than changing frozen evidence |

## Remaining public-release limitations

- A clean clone cannot independently reproduce the full GPU run without official downloads and the study's local intermediate checkpoints/outputs. The checked-in frozen manifests expose hashes and paths, not model weights.
- Raw rollout videos and logs are intentionally Git-ignored. A curated, attribution-checked video pair could be added later from **existing completed attempts** if desired; no new evaluation is needed.
- Several older documents intentionally preserve their historical open questions. The root README and new top-level documents are the authoritative current reading path.
- This study covers one LIBERO task, 17 valid final state IDs and two final unseen paraphrases. It should not be described as a full LIBERO benchmark or a demonstrated general robustness solution.
- No remote GitHub push is part of this documentation review pass.
