"""Development-only end-to-end runtime smoke test.

Strictly complies with protocol rule:
- Uses ONLY development state (state 0) and development instruction.
- Never touches or steps through held-out states 3-22.
- Never queries held-out task success.
- Validates that model policy loading, observation preprocessing, inference,
  and environment stepping execute without runtime exceptions.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _batch_obs(obs):
    """Ensure raw observation from single Gym environment has batch dimension expected by vector processors."""
    if isinstance(obs, dict):
        return {k: _batch_obs(v) for k, v in obs.items()}
    if isinstance(obs, np.ndarray):
        return np.expand_dims(obs, axis=0)
    return [obs]


class DevelopmentRuntimeSmokeTest(unittest.TestCase):
    runtime = None
    env = None
    task_config = None
    dev_instruction = None
    env_seed = 0

    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            return

        from lerobot.envs.libero import LiberoEnv
        from lerobot.utils.random_utils import set_seed
        from libero.libero import benchmark
        from smolvla_grounding.benchmark_runtime import BenchmarkRuntime

        baseline_identity = json.loads((ROOT / "outputs/language_pilot/identity.json").read_text(encoding="utf-8"))

        # Strictly use development state 0 and development instruction 0
        dev_state_idx = 0
        assert dev_state_idx == 0, "Must strictly use development state 0"
        assert dev_state_idx not in range(3, 23), "State 0 must not be in held-out states 3-22"

        cls.dev_instruction = baseline_identity["instructions"][0]
        assert cls.dev_instruction["id"] == "original"

        cls.task_config = deepcopy(baseline_identity["config"])
        cls.task_config["init_state_index"] = dev_state_idx
        cls.task_config["environment"]["episode_length"] = 280

        baseline_copy = deepcopy(baseline_identity)
        baseline_copy["config"] = cls.task_config
        baseline_copy["instructions"] = baseline_identity["instructions"]

        # Initialize BenchmarkRuntime in development/pilot scope
        cls.runtime = BenchmarkRuntime(ROOT, baseline_copy)
        load_report = cls.runtime.load_policy()
        assert load_report["strict_checkpoint_load"]

        suite_name = cls.task_config["suite"]
        suite = benchmark.get_benchmark_dict()[suite_name]()
        env_cfg = cls.task_config["environment"]
        cls.env_seed = cls.task_config.get("environment_seed", env_cfg.get("environment_seed", 0))

        set_seed(cls.env_seed)
        cls.env = LiberoEnv(
            task_suite=suite,
            task_id=cls.task_config["task_id"],
            task_suite_name=suite_name,
            init_states=True,
            episode_index=dev_state_idx,
            n_envs=1,
            obs_type=env_cfg["obs_type"],
            observation_height=env_cfg["observation_height"],
            observation_width=env_cfg["observation_width"],
            control_freq=env_cfg["control_freq"],
            control_mode=env_cfg["control_mode"],
            hard_reset=env_cfg["hard_reset"],
            num_steps_wait=env_cfg["num_steps_wait"],
            episode_length=env_cfg["episode_length"]
        )

    @classmethod
    def tearDownClass(cls):
        if cls.env is not None:
            cls.env.close()

    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA GPU not available for runtime smoke test")

    def test_development_runtime_two_step_smoke(self):
        """End-to-end execution of policy loading, 9-stage pipeline, and two consecutive environment steps on State 0."""
        import torch
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION
        from lerobot.utils.random_utils import set_seed

        # Explicit verification: only State 0 is used, never 3-22
        state_idx = self.task_config["init_state_index"]
        self.assertEqual(state_idx, 0)
        self.assertNotIn(state_idx, range(3, 23), "Held-out states 3-22 must strictly never be executed in development tests!")

        self.runtime.policy.reset()
        set_seed(self.task_config["policy_seed"])

        # Reset environment with fixed seed
        obs, _ = self.env.reset(seed=self.env_seed)

        curr_obs = obs
        for step in range(1, 3):  # Exactly two consecutive steps
            # Stage 1: Verify raw observation from env.reset() / env.step()
            self.assertIsInstance(curr_obs, dict)
            self.assertIn("pixels", curr_obs)
            self.assertIn("image", curr_obs["pixels"])
            self.assertIn("image2", curr_obs["pixels"])
            self.assertIn("robot_state", curr_obs)
            self.assertEqual(curr_obs["pixels"]["image"].shape, (256, 256, 3))
            self.assertEqual(curr_obs["pixels"]["image2"].shape, (256, 256, 3))
            self.assertIn("eef", curr_obs["robot_state"])
            self.assertEqual(curr_obs["robot_state"]["eef"]["pos"].shape, (3,))
            self.assertEqual(curr_obs["robot_state"]["eef"]["quat"].shape, (4,))

            # Stage 2: LeRobot preprocess_observation(...)
            obs_batched = _batch_obs(curr_obs)
            obs_converted = preprocess_observation(obs_batched)
            self.assertIn("observation.images.image", obs_converted)
            self.assertIn("observation.images.image2", obs_converted)
            self.assertIn("observation.robot_state", obs_converted)
            self.assertEqual(obs_converted["observation.images.image"].shape, (1, 3, 256, 256))
            self.assertEqual(obs_converted["observation.images.image2"].shape, (1, 3, 256, 256))
            self.assertEqual(obs_converted["observation.robot_state"]["eef"]["quat"].shape, (1, 4))

            # Stage 3: Attach frozen task/language instruction
            obs_converted["task"] = [self.dev_instruction["text"]]

            # Stage 4: Apply env_preprocessor (flips image channels, constructs observation.state from robot_state)
            obs_env = self.runtime.env_pre(obs_converted)
            self.assertIn("observation.images.image", obs_env)
            self.assertIn("observation.images.image2", obs_env)
            self.assertIn("observation.state", obs_env)
            self.assertIn("task", obs_env)
            self.assertEqual(obs_env["observation.state"].shape, (1, 8))

            # Stage 5: Apply policy-specific preprocessor (tokenization, normalizer, CUDA device)
            batch = self.runtime.pre(obs_env)
            self.assertIn("observation.language.tokens", batch)
            self.assertIn("observation.language.attention_mask", batch)
            self.assertEqual(batch["observation.images.image"].device.type, "cuda")
            self.assertEqual(batch["observation.state"].device.type, "cuda")
            self.assertEqual(batch["observation.state"].shape, (1, 8))

            # Verify frozen instruction tokens reached policy correctly
            tokens = batch["observation.language.tokens"][0]
            mask = batch["observation.language.attention_mask"][0].bool()
            self.assertEqual(
                tokens[mask].cpu().tolist(),
                self.runtime.token_ids[self.dev_instruction["id"]],
                "Frozen development instruction tokens must strictly match validated tokenizer ids"
            )

            # Stage 6: Policy inference
            with torch.inference_mode():
                action = self.runtime.policy.select_action(batch)
            self.assertEqual(action.shape, (1, 7))
            self.assertEqual(action.device.type, "cuda")

            # Stage 7: Apply policy-specific postprocessor (unnormalizer, CPU device)
            unnormalized_action = self.runtime.post(action)
            self.assertEqual(unnormalized_action.shape, (1, 7))
            self.assertEqual(unnormalized_action.device.type, "cpu")

            # Stage 8: Apply env_postprocessor
            action_transition = self.runtime.env_post({ACTION: unnormalized_action})
            final_action = action_transition[ACTION]

            # Stage 9: Convert to environment-compatible representation and step env
            action_np = final_action.squeeze(0).detach().cpu().numpy()
            self.assertEqual(action_np.shape, (7,))
            self.assertTrue(np.all(np.isfinite(action_np)), "Action numpy array must contain finite values")

            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            self.assertIsNotNone(next_obs)
            self.assertIn("pixels", next_obs)
            self.assertIn("robot_state", next_obs)

            curr_obs = next_obs

    def test_raw_gym_observation_to_env_preprocessor_regression(self):
        """Regression test: passing raw Gym observation to env_preprocessor fails; passing preprocessed succeeds."""
        from lerobot.envs.utils import preprocess_observation

        # Reset environment with State 0
        raw_obs, _ = self.env.reset(seed=self.env_seed)
        self.assertIsInstance(raw_obs, dict)
        self.assertIn("pixels", raw_obs)

        # 1. Reproduce original failure: raw gym observation missing "observation." prefix
        with self.assertRaises(ValueError) as ctx:
            self.runtime.env_pre(raw_obs)
        self.assertIn("ObservationProcessorStep requires an observation in the transition", str(ctx.exception))

        # 2. Demonstrate that official preprocessing path succeeds
        obs_batched = _batch_obs(raw_obs)
        obs_converted = preprocess_observation(obs_batched)
        obs_converted["task"] = [self.dev_instruction["text"]]
        obs_env = self.runtime.env_pre(obs_converted)
        self.assertIn("observation.images.image", obs_env)
        self.assertIn("observation.state", obs_env)
        self.assertEqual(obs_env["observation.state"].shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
