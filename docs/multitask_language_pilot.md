# Multi-task language sensitivity development pilot

## Frozen design

- Model: `untuned_fp32` only; no training.
- Eligible tasks: `libero_spatial/1`, `libero_spatial/4`, `libero_object/0`, `libero_goal/5`, `libero_goal/7`. `libero_goal/0` failed the 3/5 canonical competence gate.
- States: 5–9 for every task. States 30–49 are reserved and excluded.
- Each task/state uses its canonical instruction and the two fixed paraphrases below: 5 × 5 × 3 = 75 rollouts.
- Worker 1; environment/policy seeds 0; official LeRobot pipeline; 280 steps for spatial/object and 300 for goal; `goal_ever` is primary.
- All text and SHA256 values were frozen before any pilot rollout. The review below is project-author semantic review, not independent human validation.

## Frozen instruction catalog and semantic review

| Task | ID | Exact UTF-8 text | SHA256 | Semantic review |
| --- | --- | --- | --- | --- |
| `libero_spatial:1` | `original` | pick up the black bowl next to the ramekin and place it on the plate | `62903ee73ba3e6e73ed6183c529a34bf1cca052a1152e9fa6e60954b1eb7c5f3` | Keeps the black bowl adjacent to the ramekin as the target and the plate as destination. |
| `libero_spatial:1` | `paraphrase_01` | place on the plate the black bowl that is beside the ramekin | `7c4719a176a581ccf9933766b11efc6f8542fa7c62e9830a1a997361ab9ceb60` | Keeps the black bowl adjacent to the ramekin as the target and the plate as destination. |
| `libero_spatial:1` | `paraphrase_02` | take the black bowl positioned next to the ramekin, then set it on the plate | `4691692916ec3adb84b244f44c6388d79871203b667b524b287235b724ad8afc` | Keeps the black bowl adjacent to the ramekin as the target and the plate as destination. |
| `libero_spatial:4` | `original` | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | `98fef9381e261cbac30d4c58daa38fda85460c0bf0d4dd838997f741be85c3fd` | Keeps the black bowl inside the wooden cabinet top drawer and the plate as destination. |
| `libero_spatial:4` | `paraphrase_01` | from the wooden cabinet's top drawer, take the black bowl and put it on the plate | `6ea71d3820527f11fdac0ce3bde5ea41c9ed52b58ac09109d6826d13af6af05e` | Keeps the black bowl inside the wooden cabinet top drawer and the plate as destination. |
| `libero_spatial:4` | `paraphrase_02` | place the black bowl located in the top drawer of the wooden cabinet onto the plate | `4af551d231721cde809439c9472025835196b8144355fdb7145448233a465f20` | Keeps the black bowl inside the wooden cabinet top drawer and the plate as destination. |
| `libero_object:0` | `original` | pick up the alphabet soup and place it in the basket | `2d112bee564a0f6cef3403df144f91ee5c02aebe0ad96c18d47e430c2734cb54` | Keeps alphabet soup as the picked-up object and the basket as destination. |
| `libero_object:0` | `paraphrase_01` | lift the alphabet soup and put it inside the basket | `339b3130b68ec1edd57d77250eabf5369de8a8b86d56e4d632352698782b1458` | Keeps alphabet soup as the picked-up object and the basket as destination. |
| `libero_object:0` | `paraphrase_02` | place the alphabet soup in the basket after picking it up | `be088f3610755f26af010a7fa22cd36407ba394db86611745258b28e90befe63` | Keeps alphabet soup as the picked-up object and the basket as destination. |
| `libero_goal:5` | `original` | push the plate to the front of the stove | `e1bde17bec614bc5b4d0b217384eb05f91d2f1309e1822c4a72d691016a4f4aa` | Keeps pushing the plate to the front of the stove. |
| `libero_goal:5` | `paraphrase_01` | move the plate by pushing it to the front of the stove | `6663dde9e91d84456b2158b7785624ea520a753614a93a5f37001b8be2b91b9f` | Keeps pushing the plate to the front of the stove. |
| `libero_goal:5` | `paraphrase_02` | to the front of the stove, push the plate | `9de9b208838ee1c73fb60e629a81f6ce8bd90855810ae2e1c293397d2ff190da` | Keeps pushing the plate to the front of the stove. |
| `libero_goal:7` | `original` | turn on the stove | `c0d1b2f3264d13ceb9741ea5987ec2dc542ca096e0fc21573fe1b6faf2e01e31` | Keeps the goal of switching the stove on; the passive variant states the desired outcome. |
| `libero_goal:7` | `paraphrase_01` | switch the stove on | `24ee8bda0ad20f2fa1c287e4e0502665d674ceab5aeb2b6e92d1542369d12c34` | Keeps the goal of switching the stove on; the passive variant states the desired outcome. |
| `libero_goal:7` | `paraphrase_02` | the stove should be turned on | `5493b4e935632c5740dc4b8b9bf8ece5fe1ae48f6e9116130a94ab144ca1d41c` | Keeps the goal of switching the stove on; the passive variant states the desired outcome. |

Each task has one canonical sentence and two meaning-preserving variants. The variants vary word order, voice, or action wording while retaining the official referent and goal. No pilot result was used to select them. For short Goal/7, syntactic variation is necessarily limited.

## Pairing and interpretation rules

- The official initial-state asset and row index are hashed in the manifest. A completed three-prompt group must have identical captured initial observation/simulator hashes and identical Step-1 RNG-state and sampled flow-noise hashes; otherwise execution stops.
- Actual Step-1 Gaussian noise is hash-compared across prompts. Later random streams may differ after closed-loop observations and policy paths diverge.
- Action dimensions are normalized by `max(P95−P5, 0.05)` measured on all 30 canonical competence-screen trajectories (states 0–4), fixed by analysis code before pilot outcomes.
- Report every task, every prompt, and all 50 canonical/paraphrase pairs. Compare Step-1 and steps 1–10 action differences, synchronized EEF displacement, and descriptive success associations.
- EEF/action traces alone do not establish object contact or failure phase. Unsupported classifications are `unknown`. Trajectory differences after feedback are not a controlled language-only effect.
- Five states per task make all outcome contrasts descriptive; no post hoc significance threshold is used.

## Reproduction artifacts

- Frozen matrix: `configs/multitask_evaluation/language_pilot_matrix.json`.
- Frozen hash manifest: `configs/multitask_evaluation/language_pilot_manifest.json`.
- Raw local rollouts, videos, trajectories, and logs: `outputs/multitask_evaluation/runs/multitask_language_sensitivity_pilot/` (Git-ignored).
- Run entry: `scripts/run_multitask_language_pilot.py`. Analysis: `scripts/analyze_multitask_language_pilot.py`. Curated summary and figures will be added after all 75 commits pass integrity checks.
