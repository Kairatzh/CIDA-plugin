"""Тесты Этапа 0: данные судоку, CIDA-Loop, метрики и короткий прогон обучения."""

import json

import numpy as np
import pytest
import torch

from kairos_loop.metrics import auroc, expected_calibration_error
from kairos_loop.model import KairosLoop, LoopConfig, count_params
from kairos_loop.sudoku import augment, count_solutions, generate_dataset, is_valid_solution
from kairos_loop.train import parse_args, train

TINY = dict(d_model=32, n_latent=2, n_cycles=2, n_sup=3, n_agents=3, d_comm=16)


@pytest.fixture(scope="module")
def data():
    return generate_dataset(4, seed=1, min_clues=30)


class TestSudoku:
    def test_puzzles_have_unique_consistent_solution(self, data):
        puzzles, solutions = data
        for p, s in zip(puzzles, solutions):
            assert is_valid_solution(s)
            assert count_solutions(p) == 1
            given = p > 0
            assert np.array_equal(p[given], s[given])

    def test_augment_preserves_validity(self, data):
        puzzles, solutions = data
        p, s = augment(puzzles, solutions, np.random.default_rng(0))
        for pa, sa, p0 in zip(p, s, puzzles):
            assert is_valid_solution(sa)
            assert np.array_equal(pa[pa > 0], sa[pa > 0])
            assert (pa > 0).sum() == (p0 > 0).sum()
            assert count_solutions(pa) == 1


class TestModel:
    @pytest.mark.parametrize("variant,mixer,comm,bias", [
        ("plain", "mlp", True, True),
        ("cida", "mlp", True, True),
        ("cida", "attn", True, True),
        ("cida", "mlp", True, False),
        ("cida", "mlp", False, True),
    ])
    def test_step_shapes_and_finite(self, data, variant, mixer, comm, bias):
        cfg = LoopConfig(variant=variant, mixer=mixer, communicate=comm, disagreement_bias=bias, **TINY)
        model = KairosLoop(cfg)
        tokens = torch.as_tensor(data[0])
        state = model.initial_state(len(tokens))
        logits, halt, agree, (y, z) = model(tokens, state)
        assert logits.shape == (4, 81, 9)
        assert halt.shape == (4,)
        assert torch.isfinite(logits).all() and torch.isfinite(halt).all()
        assert not y.requires_grad and not z.requires_grad
        if variant == "cida":
            assert z.shape == (4, cfg.n_agents, 81, cfg.d_model)
            assert agree.shape == (4,) and ((agree >= 0) & (agree <= 1)).all()
        else:
            assert agree is None

    def test_gradients_reach_communication(self, data):
        model = KairosLoop(LoopConfig(variant="cida", **TINY))
        tokens = torch.as_tensor(data[0])
        logits, halt, _, _ = model(tokens, model.initial_state(4))
        (logits.sum() + halt.sum()).backward()
        # o обнулён на старте, но его градиент должен быть ненулевым — связь обучаема
        assert model.comm.o.weight.grad.abs().sum() > 0
        assert model.roles.grad is not None

    def test_cida_adds_few_params(self):
        plain = count_params(KairosLoop(LoopConfig(variant="plain")))
        cida = count_params(KairosLoop(LoopConfig(variant="cida")))
        assert 0 < cida - plain < 0.1 * plain

    def test_invalid_config(self):
        with pytest.raises(ValueError):
            LoopConfig(variant="cida", n_agents=1)


class TestMetrics:
    def test_auroc(self):
        assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
        assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
        assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
        assert np.isnan(auroc([0.1, 0.2], [1, 1]))

    def test_ece(self):
        assert expected_calibration_error([1.0, 1.0], [1, 1]) == pytest.approx(0.0)
        assert expected_calibration_error([0.9] * 10, [1] * 5 + [0] * 5) == pytest.approx(0.4)


def test_train_smoke(tmp_path):
    args = parse_args([
        "--variant", "cida", "--data", "generated", "--n-train", "4", "--n-test", "2", "--min-clues", "35",
        "--d-model", "32", "--n-latent", "2", "--n-cycles", "2", "--n-sup", "2", "--n-agents", "2",
        "--d-comm", "16", "--iters", "3", "--batch-size", "4", "--eval-every", "3", "--log-every", "1",
        "--warmup", "1", "--device", "cpu", "--out", str(tmp_path / "run"),
    ])
    result = train(args)
    assert np.isfinite(result["final"]["cell_acc"])
    saved = json.loads((tmp_path / "run" / "result.json").read_text())
    assert saved["config"]["variant"] == "cida"
    assert (tmp_path / "run" / "model.pt").exists()
