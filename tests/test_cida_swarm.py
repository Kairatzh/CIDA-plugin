"""Тесты CIDA-Swarm: среда поиска, мозг дрона, критик и короткий прогон MAPPO."""

import json

import pytest
import torch

from cida_swarm.env import DET_FEATURES, N_ACTIONS, SELF_FEATURES, EnvConfig, SearchEnv
from cida_swarm.policy import Actor, Critic
from cida_swarm.train import evaluate, parse_args, train

SMALL = dict(n_envs=8, n_drones=4, min_drones=2, n_targets=3, n_decoys=2, grid=12, max_steps=20)


def make_env(**kw):
    return SearchEnv(EnvConfig(**{**SMALL, **kw}), seed=0)


def stay(env):
    return torch.zeros(env.cfg.n_envs, env.cfg.n_drones, dtype=torch.long)


class TestEnv:
    def test_shapes_and_bounds(self):
        env = make_env()
        obs = env.reset()
        assert obs["self"].shape == (8, 4, SELF_FEATURES)
        assert obs["det"].shape == (8, 4, 6, DET_FEATURES)
        assert obs["cov"].shape == (8, 4, 144) and obs["cov"].dtype == torch.bool
        assert obs["obstacles"].shape == (8, 144)
        assert obs["cell"].shape == (8, 4) and obs["active"].shape == (8, 4)
        assert env.state().shape == (8, SearchEnv.state_dim(env.cfg))
        for _ in range(30):
            obs, reward, done, _ = env.step(torch.randint(0, N_ACTIONS, (8, 4)))
            assert ((env.pos >= 0) & (env.pos <= 1)).all()
            assert reward.shape == (8,) and done.shape == (8,)

    def test_active_drone_count_in_range(self):
        env = make_env(n_envs=256)
        env.reset()
        n = env.active.sum(1)
        assert (n >= 2).all() and (n <= 4).all() and n.unique().numel() > 1
        assert env.active[:, 0].all()  # активные — первые по индексу

    def test_drones_never_inside_obstacles(self):
        env = make_env(n_envs=64, n_obstacles=8, obstacle_min=3, obstacle_max=5)
        env.reset()
        for _ in range(40):
            env.step(torch.randint(0, N_ACTIONS, (64, 4)))
            inside = env._blocked(env.pos, env.obstacles) & env.active
            assert not inside.any()

    def test_people_spawn_outside_obstacles(self):
        env = make_env(n_envs=64, n_obstacles=8, obstacle_min=3, obstacle_max=5)
        env.reset()
        assert not env._blocked(env.targets, env.obstacles).any()

    def test_finding_target_rewards_and_removes_it(self):
        env = make_env(reward_explore=0.0, target_speed=0.0)
        env.reset()
        env.targets[0, 0] = env.pos[0, 0]
        env.vel.zero_()
        _, reward, _, _ = env.step(stay(env))
        assert reward[0] >= 1.0
        assert not env.alive[0, 0]

    def test_inactive_drones_do_not_find_or_move(self):
        env = make_env(reward_explore=0.0, target_speed=0.0)
        env.reset()
        env.active[0] = torch.tensor([True, False, False, False])
        env.targets[0, 0] = env.pos[0, 3]
        before = env.pos[0, 3].clone()
        env.step(torch.ones(8, 4, dtype=torch.long))
        assert env.alive[0, 0]
        assert torch.equal(env.pos[0, 3], before)

    def test_episode_ends_and_resets(self):
        env = make_env(max_steps=5)
        env.reset()
        for _ in range(5):
            _, _, done, info = env.step(stay(env))
        assert done.all()
        assert info["length"].eq(5).all()
        assert env.t.eq(0).all() and env.alive.all()

    def test_reset_only_touches_done_worlds(self):
        env = make_env()
        env.reset()
        env.step(stay(env))
        pos_before = env.pos.clone()
        mask = torch.zeros(8, dtype=torch.bool)
        mask[0] = True
        env._reset_where(mask)
        assert torch.equal(env.pos[1:], pos_before[1:])
        assert env.t[0] == 0 and env.t[1] == 1

    def test_all_found_ends_episode(self):
        env = make_env(n_targets=1, target_speed=0.0)
        env.reset()
        env.targets[:, 0] = env.pos[:, 0]
        _, _, done, info = env.step(stay(env))
        assert done.all() and info["found_frac"].eq(1.0).all()

    def test_detection_only_within_view(self):
        env = make_env(p_detect=1.0, false_pos_rate=0.0, pos_noise=0.0, n_decoys=0, n_obstacles=0)
        env.reset()
        env.active[:] = True
        env.pos[:] = 0.1
        env.targets[:, 0] = 0.11   # вплотную — видно почти всегда
        env.targets[:, 1:] = 0.9   # далеко — не видно никогда
        seen = torch.stack([env.observe()["det"][..., -1].sum(-1) for _ in range(20)])
        assert seen.le(1).all()
        assert seen.mean() > 0.8

    def test_decoys_are_visible_but_not_findable(self):
        env = make_env(p_detect=1.0, false_pos_rate=0.0, n_targets=1, n_decoys=1, n_obstacles=0, target_speed=0.0)
        env.reset()
        env.active[:] = True
        env.pos[:] = 0.5
        env.targets[:, 0] = 0.95
        env.decoys[:, 0] = 0.5
        assert (env.observe()["det"][..., -1].sum(-1) >= 1).all()
        env.step(stay(env))
        assert env.alive.all()

    def test_shared_coverage_is_team_map(self):
        env = make_env(share_coverage=True)
        obs = env.reset()
        assert torch.equal(obs["cov"][:, 0], obs["cov"][:, 1])

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            EnvConfig(n_drones=3, min_drones=4)


class TestPolicy:
    def test_actor_critic_shapes(self):
        env = make_env()
        obs = env.reset()
        actor, critic = Actor(env.cfg, hidden=32), Critic(env.cfg, hidden=32)
        assert actor(obs).shape == (8, 4, N_ACTIONS)
        assert critic(env.state()).shape == (8,)

    def test_actor_handles_time_batch(self):
        env = make_env()
        o1 = env.reset()
        o2, *_ = env.step(stay(env))
        stacked = {k: torch.stack([o1[k], o2[k]]) for k in o1}
        actor = Actor(env.cfg, hidden=32)
        assert actor(stacked).shape == (2, 8, 4, N_ACTIONS)

    def test_actor_ignores_detection_order(self):
        env = make_env(p_detect=1.0, false_pos_rate=0.0)
        obs = env.reset()
        actor = Actor(env.cfg, hidden=32)
        flipped = dict(obs, det=obs["det"].flip(-2))
        assert torch.allclose(actor(obs), actor(flipped), atol=1e-5)


def test_random_eval_runs():
    m = evaluate(None, EnvConfig(**SMALL), n_envs=8, device="cpu")
    assert 0.0 <= m["found_frac"] <= 1.0 and 1 <= m["length"] <= 20
    assert 0.0 < m["coverage"] <= 1.0


def test_train_smoke(tmp_path):
    args = parse_args([
        "--n-envs", "8", "--n-drones", "4", "--min-drones", "2", "--n-targets", "3", "--grid", "12",
        "--max-steps", "20", "--hidden", "16", "--iters", "3", "--rollout", "16", "--epochs", "1",
        "--minibatch-size", "64", "--log-every", "1", "--eval-every", "3", "--eval-envs", "8",
        "--device", "cpu", "--out", str(tmp_path / "run"),
    ])
    result = train(args)
    assert 0.0 <= result["final"]["found_frac"] <= 1.0
    assert result["agent_steps_per_s"] > 0
    saved = json.loads((tmp_path / "run" / "result.json").read_text())
    assert "random" in saved and (tmp_path / "run" / "model.pt").exists()
