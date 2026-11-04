"""Тесты этапа 4: леса подсказок, банк трудных карт, арена «спасатели против охотников» и её обучение."""

import torch

from cida_swarm.arena import ArenaEnv
from cida_swarm.rescue import N_ACTIONS, RescueConfig, RescueEnv
from cida_swarm.train import parse_args, train

STAY, PX, MX, PY, MY = range(5)


def arena(**kw):
    cfg = RescueConfig(**{**dict(n_envs=4, n_drones=3, min_drones=3, n_victims=1, n_hunters=1,
                                 difficulty_min=1, difficulty_max=1), **kw})
    env = ArenaEnv(cfg, seed=0)
    env.reset()
    env.q_active[:] = True
    return env


def test_hinted_world_gives_the_base_swarm_its_old_inputs():
    a = RescueEnv(RescueConfig(n_envs=16, share_map=True), seed=3)
    b = RescueEnv(RescueConfig(n_envs=16, route_hint=False, hidden_weight=True), seed=3)
    b.hint_prob = 1.0
    oa, ob = a.reset(), b.reset()
    for _ in range(20):
        assert all(torch.equal(oa[k], ob[k]) for k in ("window", "coarse", "self"))
        act = torch.randint(0, N_ACTIONS, (16, 6))
        oa, *_ = a.step(act)
        ob, *_ = b.step(act)


def test_unhinted_world_hides_the_route():
    env = RescueEnv(RescueConfig(n_envs=16, route_hint=False), seed=1)
    w = env.reset()["window"]
    alive = env.alive[..., None]
    assert ((w[:, :, 8] == 251) | ~alive).all()


def test_hard_bank_collects_failures_and_serves_them():
    env = RescueEnv(RescueConfig(n_envs=32, n_drones=3, min_drones=3, max_steps=5, bank_size=64, bank_frac=1.0), seed=0)
    env.reset()
    for _ in range(12):
        env.step(torch.zeros(32, 3, dtype=torch.long))
    assert int(env.bank_count) > 0
    env._refresh_pool()
    assert env._fresh["walls"].shape == (32, env.cells)


def test_hunter_catches_adjacent_rescuer_outside_entrance():
    env = arena()
    env.pos[0] = torch.tensor([[6, 2], [13, 13], [13, 9]])
    env.qpos[0, 0] = torch.tensor([7, 2])
    env.step(torch.zeros(4, 3, dtype=torch.long), torch.zeros(4, 1, dtype=torch.long))
    assert not env.alive[0, 0] and env.alive[0, 1]
    assert env.hunter_reward[0] >= env.cfg.reward_catch - 0.1


def test_two_rescuers_stun_the_hunter():
    env = arena()
    env.pos[0] = torch.tensor([[6, 2], [8, 2], [13, 13]])
    env.qpos[0, 0] = torch.tensor([7, 2])
    _, reward, _, _ = env.step(torch.zeros(4, 3, dtype=torch.long), torch.zeros(4, 1, dtype=torch.long))
    assert env.q_stun[0, 0] > 0 and env.alive[0, :2].all()          # оглушён раньше, чем поймал
    assert reward[0] > 0


def test_hunters_stay_out_of_the_entrance_and_walls():
    env = arena(n_envs=16, n_hunters=3, difficulty_max=4)
    for _ in range(200):
        env.step(torch.randint(0, N_ACTIONS, (16, 3)), torch.randint(0, N_ACTIONS, (16, 3)))
        idx = env._idx(env.qpos)
        assert not env.walls.gather(1, idx).any()
        assert not (env.exit_mask[idx] & env.q_active).any()


def test_rescuers_see_hunters_as_danger():
    env = arena()
    env.pos[0, 0] = torch.tensor([6, 2])
    env.qpos[0, 0] = torch.tensor([8, 2])
    w = env.observe()["window"][0, 0, 3].view(7, 7)
    assert w[3 + 2, 3] == 1


def test_arena_self_play_trains_and_saves_both_teams(tmp_path):
    small = ["--world", "rescue", "--n-envs", "8", "--n-drones", "3", "--min-drones", "2", "--n-victims", "2",
             "--max-steps", "30", "--hidden", "16", "--rollout", "8", "--epochs", "1", "--minibatch-size", "32",
             "--log-every", "1", "--eval-envs", "8", "--device", "cpu"]
    train(parse_args(small + ["--iters", "1", "--eval-every", "1", "--out", str(tmp_path / "base")]))
    result = train(parse_args(small + ["--hunters", "2", "--memory", "--personas", "--roles", "--comm", "cida",
                                       "--hard-bank", "32", "--freeze-base", "--league-every", "1", "--league-prob", "0.5",
                                       "--init-from", str(tmp_path / "base" / "model.pt"), "--iters", "4",
                                       "--eval-every", "2", "--out", str(tmp_path / "arena")]))
    assert 0.0 <= result["final"]["rescued_frac"] <= 1.0 and "vs_first_hunters" in result["final"]
    ckpt = torch.load(tmp_path / "arena" / "model.pt", weights_only=False)
    assert ckpt["world"] == "arena" and "hunter" in ckpt


def test_arena_demo_shows_hunters(tmp_path):
    test_arena_self_play_trains_and_saves_both_teams(tmp_path)
    from cida_swarm.demo import build, write_html
    data = build([str(tmp_path / "arena" / "model_best.pt")], ["CIDA против охотников"], seeds=[1], difficulties=[3],
                 n_drones=3, greedy=False)
    frames = data["scenes"][0]["panels"][0]["episode"]["frames"]
    assert frames[0]["q"] and len(frames[0]["q"][0]) == 3
    write_html(data, str(tmp_path / "arena.html"))
    assert "--hunter" in (tmp_path / "arena.html").read_text(encoding="utf-8")
