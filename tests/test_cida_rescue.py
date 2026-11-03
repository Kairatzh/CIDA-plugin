"""Тесты CIDA-Rescue: здание, переноска вдвоём, завалы, пожар, доставка, мозг и обучение."""

from collections import deque

import pytest
import torch

from cida_swarm.policy import RescueActor
from cida_swarm.rescue import INF, N_ACTIONS, SELF_FEATURES, RescueConfig, RescueEnv
from cida_swarm.train import evaluate, parse_args, train

STAY, PX, MX, PY, MY = range(5)
RELEASE = 5


def make_env(**kw):
    cfg = RescueConfig(**{**dict(n_envs=4, n_drones=3, min_drones=3, n_victims=1, difficulty_min=1,
                                 difficulty_max=1), **kw})
    env = RescueEnv(cfg, seed=0)
    env.reset()
    return env


def place(env, world, drones=(), victim=None, weight=1):
    """Ставит дронов и пострадавшего в заданные клетки мира world."""
    for k, p in enumerate(drones):
        env.pos[world, k] = torch.tensor(p)
    if victim is not None:
        env.vpos[world, 0] = torch.tensor(victim)
        env.vweight[world, 0] = weight
        env.present[world, 0] = True


def act(env, world, per_drone):
    a = torch.zeros(env.cfg.n_envs, env.cfg.n_drones, dtype=torch.long)
    for k, v in enumerate(per_drone):
        a[world, k] = v
    return env.step(a)


def bfs(walls, rubble, g):
    dist = [[INF] * g for _ in range(g)]
    q = deque()
    for x in range(1, 4):
        for y in range(1, 4):
            dist[x][y] = 0
            q.append((x, y))
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < g and 0 <= ny < g and not walls[nx * g + ny] and rubble[nx * g + ny] == 0 \
                    and dist[nx][ny] > dist[x][y] + 1:
                dist[nx][ny] = dist[x][y] + 1
                q.append((nx, ny))
    return dist


class TestBuilding:
    def test_shapes(self):
        env = make_env(n_victims=2)
        obs = env.observe()
        assert obs["window"].shape == (4, 3, 9, 49) and obs["window"].dtype == torch.uint8
        assert obs["coarse"].shape == (4, 3, 2, 49)
        assert obs["self"].shape == (4, 3, SELF_FEATURES)
        assert env.state().shape == (4, RescueEnv.state_dim(env.cfg))
        assert env.belief_target().shape == (4, 49)

    def test_buildings_are_connected_without_rubble(self):
        env = make_env(n_envs=64, difficulty_min=1, difficulty_max=2)
        interior = env.interior
        assert (env.dist[:, interior] < INF).all()   # из любой комнаты есть путь к выходу

    def test_distance_matches_bfs(self):
        env = make_env(n_envs=8, difficulty_min=3, difficulty_max=3)
        g = env.G
        for w in range(8):
            ref = bfs(env.walls[w].tolist(), env.rubble[w].tolist(), g)
            got = env.dist[w].view(g, g)
            for x in range(g):
                for y in range(g):
                    assert got[x, y].item() == pytest.approx(ref[x][y]), (w, x, y)

    def test_drones_start_in_exit_room_and_victims_elsewhere(self):
        env = make_env(n_envs=32, n_victims=3)
        assert env.exit_mask[env._idx(env.pos)].all()
        assert not env.exit_mask[env._idx(env.vpos)].any()

    def test_difficulty_controls_content(self):
        easy = make_env(n_envs=64, n_victims=3, difficulty_min=1, difficulty_max=1)
        assert easy.vweight.eq(1).all() and easy.rubble.eq(0).all() and not easy.hazard_on.any()
        hard = make_env(n_envs=64, n_victims=3, difficulty_min=4, difficulty_max=4)
        assert hard.vweight.max() > 1 and hard.rubble.gt(0).any() and hard.hazard_on.all()


class TestMechanics:
    def test_heavy_victim_needs_enough_carriers(self):
        env = make_env()
        place(env, 0, drones=[(6, 2), (6, 2), (10, 10)], victim=(6, 2), weight=2)
        act(env, 0, [MX, STAY, STAY])
        assert env.vpos[0, 0].tolist() == [6, 2]          # один не сдвинет
        assert env.last["waiting"][0, 0]
        act(env, 0, [MX, MX, STAY])
        assert env.vpos[0, 0].tolist() == [5, 2]          # вдвоём — да
        assert env.pos[0, 0].tolist() == [5, 2] and env.pos[0, 1].tolist() == [5, 2]

    def test_released_victim_stays(self):
        env = make_env()
        place(env, 0, drones=[(6, 2), (10, 10), (10, 10)], victim=(6, 2), weight=1)
        act(env, 0, [RELEASE + MX, STAY, STAY])
        assert env.vpos[0, 0].tolist() == [6, 2] and env.pos[0, 0].tolist() == [5, 2]

    def test_carriers_must_agree_on_direction(self):
        env = make_env()
        place(env, 0, drones=[(6, 2), (6, 2), (10, 10)], victim=(6, 2), weight=2)
        act(env, 0, [MX, PY, STAY])
        assert env.vpos[0, 0].tolist() == [6, 2]

    def test_delivery_rewards_and_ends(self):
        env = make_env()
        env.walls[0, env._idx(torch.tensor([4, 2]))] = False
        place(env, 0, drones=[(4, 2), (10, 10), (10, 10)], victim=(4, 2), weight=1)
        _, reward, done, info = act(env, 0, [MX, STAY, STAY])
        assert reward[0] >= env.cfg.reward_deliver
        assert done[0] and info["rescued_frac"][0] == 1.0

    def test_rubble_needs_enough_pushers(self):
        env = make_env()
        cell = env._idx(torch.tensor([5, 2]))
        env.rubble[0, cell] = 2
        place(env, 0, drones=[(6, 2), (6, 2), (10, 10)], victim=(13, 13))
        act(env, 0, [MX, STAY, STAY])
        assert env.rubble[0, cell] == 2 and env.pos[0, 0].tolist() == [6, 2]
        _, reward, _, _ = act(env, 0, [MX, MX, STAY])
        assert env.rubble[0, cell] == 0 and reward[0] >= env.cfg.reward_clear

    def test_walls_block_movement(self):
        env = make_env()
        place(env, 0, drones=[(1, 1), (2, 2), (2, 2)], victim=(13, 13))
        act(env, 0, [MX, STAY, STAY])
        assert env.pos[0, 0].tolist() == [1, 1]

    def test_hazard_burns_nearby_drones(self):
        env = make_env(hazard_move_prob=0.0)
        env.hazard_on[0] = True
        env.hpos[0] = torch.tensor([7, 2])
        place(env, 0, drones=[(6, 2), (2, 2), (2, 2)], victim=(13, 13))
        _, reward, _, info = act(env, 0, [STAY, STAY, STAY])
        assert not env.alive[0, 0] and env.alive[0, 1]
        assert reward[0] < 0

    def test_episode_ends_when_all_drones_lost(self):
        env = make_env(hazard_move_prob=0.0)
        env.hazard_on[0] = True
        env.hpos[0] = torch.tensor([2, 2])
        _, _, done, _ = act(env, 0, [STAY, STAY, STAY])
        assert done[0]

    def test_belief_target_marks_waiting_victims(self):
        env = make_env()
        place(env, 0, victim=(6, 2))
        assert env.belief_target()[0, (6 // 3) * 7 + 2 // 3]

    def test_random_play_is_stable(self):
        env = make_env(n_envs=16, n_victims=3, difficulty_min=1, difficulty_max=4)
        for _ in range(400):
            env.step(torch.randint(0, N_ACTIONS, (16, 3)))
            assert env.walls.gather(1, env._idx(env.pos)).logical_and(env.alive).logical_not().all()


class TestRescueBrain:
    @pytest.mark.parametrize("comm", ["none", "attn", "cida"])
    def test_actor_shapes(self, comm):
        env = make_env(n_victims=2)
        actor = RescueActor(env.cfg, hidden=32, comm=comm)
        logits, aux = actor(env.observe(), return_aux=True)
        assert logits.shape == (4, 3, N_ACTIONS)
        assert (aux is None) == (comm == "none")

    def test_attention_baseline_has_no_disagreement_bias(self):
        env = make_env()
        attn = RescueActor(env.cfg, hidden=16, comm="attn").comm
        cida = RescueActor(env.cfg, hidden=16, comm="cida").comm
        assert not attn.alpha.requires_grad and attn.alpha == 0
        assert cida.alpha.requires_grad

    def test_belief_loss_does_not_move_features(self):
        env = make_env()
        actor = RescueActor(env.cfg, hidden=16, comm="cida")
        _, aux = actor(env.observe(), return_aux=True)
        sum(b.sum() for b in aux["beliefs"]).backward()
        assert actor.trunk[0].weight.grad is None or actor.trunk[0].weight.grad.abs().sum() == 0


RESCUE_ARGS = ["--world", "rescue", "--n-envs", "8", "--n-drones", "3", "--min-drones", "2", "--n-victims", "2",
               "--max-steps", "30", "--hidden", "16", "--rollout", "8", "--epochs", "1", "--minibatch-size", "32",
               "--log-every", "1", "--eval-envs", "8", "--device", "cpu"]


def test_random_eval_reports_levels():
    cfg = RescueConfig(n_envs=8, n_drones=3, min_drones=2, n_victims=2, max_steps=30)
    m = evaluate(None, cfg, n_envs=16, device="cpu", world="rescue")
    assert 0.0 <= m["rescued_frac"] <= 1.0
    assert "by_level" in m and set(m["by_level"]) <= {"1", "2", "3", "4"}


def test_rescue_train_and_finetune(tmp_path):
    train(parse_args(RESCUE_ARGS + ["--share-coverage", "--iters", "2", "--eval-every", "2",
                                    "--out", str(tmp_path / "base")]))
    assert (tmp_path / "base" / "model_best.pt").exists()
    result = train(parse_args(RESCUE_ARGS + ["--share-coverage", "--comm", "cida", "--iters", "1",
                                             "--eval-every", "1", "--init-from", str(tmp_path / "base" / "model.pt"),
                                             "--out", str(tmp_path / "ft")]))
    assert 0.0 <= result["best"]["rescued_frac"] <= 1.0

    from cida_swarm.demo import build, write_html
    data = build([str(tmp_path / "base" / "model_best.pt"), str(tmp_path / "ft" / "model_best.pt")],
                 ["база", "CIDA"], seeds=[1], difficulties=[4], n_drones=3, greedy=False)
    panel = data["scenes"][0]["panels"][1]["episode"]
    assert panel["frames"] and "e" in panel["frames"][0] and "b" in panel["frames"][0]
    out = tmp_path / "demo.html"
    write_html(data, str(out))
    assert "CIDA-Rescue" in out.read_text(encoding="utf-8")
