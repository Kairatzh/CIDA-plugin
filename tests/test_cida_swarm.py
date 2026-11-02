"""Тесты шага 1 CIDA-Swarm: среда поиска, мозг дрона, критик и короткий прогон MAPPO."""

import json

import pytest
import torch

from cida_swarm.env import DET_FEATURES, N_ACTIONS, SELF_FEATURES, EnvConfig, SearchEnv
from cida_swarm.policy import Actor, Critic
from cida_swarm.train import evaluate, parse_args, train

SMALL = dict(n_envs=8, n_drones=3, n_targets=2, max_steps=20)


def make_env(**kw):
    return SearchEnv(EnvConfig(**{**SMALL, **kw}), seed=0)


class TestEnv:
    def test_shapes_and_bounds(self):
        env = make_env()
        obs = env.reset()
        assert obs["self"].shape == (8, 3, SELF_FEATURES)
        assert obs["det"].shape == (8, 3, 4, DET_FEATURES)
        assert obs["cov"].shape == (8, 3, 16 * 16)
        assert env.state().shape == (8, SearchEnv.state_dim(env.cfg))
        for _ in range(30):
            obs, reward, done, _ = env.step(torch.randint(0, N_ACTIONS, (8, 3)))
            assert ((env.pos >= 0) & (env.pos <= 1)).all()
            assert reward.shape == (8,) and done.shape == (8,)

    def test_finding_target_rewards_and_removes_it(self):
        env = make_env(reward_explore=0.0)
        env.reset()
        env.targets[0, 0] = env.pos[0, 0]
        env.vel.zero_()
        _, reward, _, _ = env.step(torch.zeros(8, 3, dtype=torch.long))
        assert reward[0] >= 1.0
        assert not env.alive[0, 0]

    def test_episode_ends_and_resets(self):
        env = make_env(max_steps=5)
        env.reset()
        for i in range(5):
            _, _, done, info = env.step(torch.zeros(8, 3, dtype=torch.long))
        assert done.all()
        assert info["length"].eq(5).all()
        assert env.t.eq(0).all() and env.alive.all()

    def test_all_found_ends_episode(self):
        env = make_env(n_targets=1)
        env.reset()
        env.targets[:, 0] = env.pos[:, 0]
        _, _, done, info = env.step(torch.zeros(8, 3, dtype=torch.long))
        assert done.all() and info["found_frac"].eq(1.0).all()

    def test_detection_only_within_view(self):
        env = make_env(p_detect=1.0, false_pos_rate=0.0, pos_noise=0.0)
        env.reset()
        env.pos[:] = 0.1
        env.targets[:, 0] = 0.15   # близко
        env.targets[:, 1] = 0.9    # далеко
        det = env.observe()["det"]
        assert det[..., -1].sum(-1).eq(1).all()

    def test_exploration_reward_counts_new_cells_once(self):
        env = make_env(reward_find=0.0)
        env.reset()
        _, r1, _, _ = env.step(torch.zeros(8, 3, dtype=torch.long))
        assert (r1 >= 0).all()
        before = env.team_cov.sum()
        env.step(torch.zeros(8, 3, dtype=torch.long))
        assert env.team_cov.sum() >= before

    def test_shared_coverage_is_team_map(self):
        env = make_env(share_coverage=True)
        obs = env.reset()
        assert torch.equal(obs["cov"][:, 0], obs["cov"][:, 1])


class TestPolicy:
    def test_actor_critic_shapes(self):
        env = make_env()
        obs = env.reset()
        actor, critic = Actor(env.cfg, hidden=32), Critic(env.cfg, hidden=32)
        assert actor(obs).shape == (8, 3, N_ACTIONS)
        assert critic(env.state()).shape == (8,)

    def test_actor_ignores_detection_order(self):
        env = make_env(p_detect=1.0, false_pos_rate=0.0, k_detect=3)
        obs = env.reset()
        actor = Actor(env.cfg, hidden=32)
        flipped = dict(obs, det=obs["det"].flip(-2))
        assert torch.allclose(actor(obs), actor(flipped), atol=1e-6)


def test_random_eval_runs():
    m = evaluate(None, EnvConfig(**SMALL), n_envs=8, device="cpu")
    assert 0.0 <= m["found_frac"] <= 1.0 and 1 <= m["length"] <= 20


def test_train_smoke(tmp_path):
    args = parse_args([
        "--n-envs", "8", "--n-drones", "3", "--n-targets", "2", "--max-steps", "20", "--hidden", "16",
        "--iters", "3", "--rollout", "16", "--epochs", "1", "--minibatches", "2", "--log-every", "1",
        "--eval-every", "3", "--eval-envs", "8", "--device", "cpu", "--out", str(tmp_path / "run"),
    ])
    result = train(args)
    assert 0.0 <= result["final"]["found_frac"] <= 1.0
    saved = json.loads((tmp_path / "run" / "result.json").read_text())
    assert "random" in saved and (tmp_path / "run" / "model.pt").exists()
