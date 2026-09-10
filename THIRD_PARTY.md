# Upstream software, models and data

This repository builds an experiment and analysis layer around official implementations.
It is not affiliated with Hugging Face or the LIBERO authors.

| Component | Role | Source |
| --- | --- | --- |
| LeRobot | SmolVLA implementation, processors, training utilities and evaluator | [huggingface/lerobot](https://github.com/huggingface/lerobot) |
| SmolVLA LIBERO checkpoint | Starting policy weights and normalization statistics | [HuggingFaceVLA/smolvla_libero](https://huggingface.co/HuggingFaceVLA/smolvla_libero) |
| SmolVLM2 | Pretrained vision-language backbone | [HuggingFaceTB/SmolVLM2-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Instruct) |
| LIBERO | Tasks, instructions, initial states, assets and goal predicates | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| Hugging Face LIBERO packaging | LeRobot-compatible environment distribution | [huggingface/LIBERO](https://github.com/huggingface/LIBERO) |
| LIBERO demonstrations | Robot demonstration data | [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero), [lerobot/libero](https://huggingface.co/datasets/lerobot/libero) |
| MuJoCo / robosuite | Physics and robot simulation | [MuJoCo](https://github.com/google-deepmind/mujoco), [robosuite](https://github.com/ARISE-Initiative/robosuite) |

The project-specific code and documentation are covered by [MIT](LICENSE).
This does not relicense upstream software, model weights, datasets, instructions,
or simulator assets. Consult the relevant upstream repository and model/dataset
cards for their terms and citation information. Simulator screenshots in `results/`
and `docs/` are generated rollout observations containing upstream scene assets.

No upstream source tree, model weights or demonstration shards are vendored here.
Downloads stay in ignored local directories. Exact model and dataset revisions are
recorded in `configs/` and the data audit under `docs/`.

Our contribution is the controlled instruction protocol, pairing and repeat
diagnostics, data-selection validation, matched language fine-tuning experiments,
resumable experiment records, and result/qualitative analysis. The policy architecture,
forward pass, loss and action sampler remain upstream implementations. A small local
loader adapter preserves FP32 trainable parameters across checkpoint reloads; it
does not introduce a new policy architecture.
