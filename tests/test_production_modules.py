"""
Unit tests for Production-Ready CIDA Platform Modules:
  - FiLM Conditioning Layer (film.py)
  - Latent MCTS Search Engine (latent_mcts.py)
  - Strategy Library & Meta-Learning (strategy_library.py)
  - CIDABreakthroughEngine Integration (breakthrough.py)
"""

import os
import pytest
import torch
import torch.nn as nn

from cida_plugin.inference.film import FiLMGenerator, FiLMModulatedGeneration
from cida_plugin.inference.latent_mcts import LatentMCTS, MCTSNode
from cida_plugin.inference.strategy_library import StrategyLibrary, Strategy, InMemoryStrategyStore
from cida_plugin.inference.breakthrough import CIDABreakthroughEngine


def test_film_generator_shapes():
    d_cida = 512
    d_llm = 64
    num_layers = 3

    film_gen = FiLMGenerator(d_cida=d_cida, d_llm=d_llm, num_film_layers=num_layers)

    # Test single-batch 2D state
    cida_state_2d = torch.randn(1, d_cida)
    film_params = film_gen(cida_state_2d)

    assert len(film_params) == num_layers
    for gamma, beta in film_params:
        assert gamma.shape == (1, d_llm)
        assert beta.shape == (1, d_llm)
        # Check initial near-identity weights (gamma ≈ 1, beta ≈ 0)
        assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-2)
        assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-2)

    # Test multi-agent 3D state flattening
    cida_state_3d = torch.randn(1, 4, 128)  # 4 agents, d_hidden 128 -> 512
    film_params_3d = film_gen(cida_state_3d)
    assert len(film_params_3d) == num_layers
    assert film_params_3d[0][0].shape == (1, d_llm)


def test_film_modulation_hooks():
    d_llm = 16

    # Create dummy multi-layer Transformer
    class DummyTransformerLayer(nn.Module):
        def forward(self, x):
            return (x + 1.0,)

    class DummyLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([DummyTransformerLayer(), DummyTransformerLayer()])
            self.param = nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])

    class DummyTokenizer:
        def __call__(self, text, return_tensors):
            class Enc:
                def __init__(self):
                    self.input_ids = torch.tensor([[10, 20]])
                    self.attention_mask = torch.tensor([[1, 1]])
                def to(self, device): return self
            return Enc()

        def decode(self, ids, skip_special_tokens=True):
            return "Modulated Output"

        @property
        def pad_token_id(self): return 0
        @property
        def eos_token_id(self): return 1

    llm = DummyLLM()
    tok = DummyTokenizer()
    film_gen = FiLMGenerator(d_cida=32, d_llm=d_llm, num_film_layers=2)

    film_mod = FiLMModulatedGeneration(llm, tok, film_gen)

    # Test hook installation and removal
    cida_state = torch.ones(1, 32)
    film_params = film_mod.get_film_params(cida_state)
    assert len(film_params) == 2

    # Install hooks manually and check layer execution
    film_mod._install_hooks(film_params)
    assert len(film_mod._hooks) == 2

    # Output tuple modification check: (x+1) modulated by gamma=1, beta=0 should equal x+1
    dummy_input = torch.zeros(1, 4, d_llm)
    out = llm.model.layers[0](dummy_input)
    assert torch.allclose(out[0], torch.ones(1, 4, d_llm), atol=1e-2)

    film_mod._remove_hooks()
    assert len(film_mod._hooks) == 0


def test_mcts_tree_construction():
    d_hidden = 16
    h_base = torch.ones(1, d_hidden)

    mcts = LatentMCTS(exploration_constant=1.414)
    root = mcts.initialize_tree(h_base)

    assert root.visit_count == 1
    assert root.value == 0.0

    # Explore branches
    children = mcts.explore_branches(root, num_branches=3, noise_scale=0.1)
    assert len(children) == 3
    assert len(root.children) == 3

    # Backpropagate score to child 0
    mcts.backpropagate(children[0], score=1.0)
    assert children[0].visit_count == 1
    assert children[0].value == 1.0
    assert root.visit_count == 2
    assert root.value_sum == 1.0


def test_mcts_ucb_selection():
    d_hidden = 16
    h_base = torch.ones(1, d_hidden)

    mcts = LatentMCTS(exploration_constant=1.0)
    root = mcts.initialize_tree(h_base)
    children = mcts.explore_branches(root, num_branches=2)

    # Unvisited children have UCB = inf
    selected = mcts.select_best_child(root)
    assert selected in children

    # Backprop child 0 with high score, child 1 with low score
    mcts.backpropagate(children[0], score=5.0)
    mcts.backpropagate(children[1], score=0.1)

    # Best optimal node selection
    best_node = mcts.select_optimal_node()
    assert best_node == children[0]


def test_mcts_steering_vector():
    d_hidden = 8
    h_base = torch.ones(1, d_hidden)

    mcts = LatentMCTS(steering_learning_rate=0.15)
    root = mcts.initialize_tree(h_base)

    # Create failed nodes away from base
    failed_h = torch.zeros(1, d_hidden)
    failed_node = MCTSNode(h=failed_h, parent=root)

    offset = mcts.compute_steering_vector(h_base, [failed_node])
    assert offset is not None
    assert offset.shape == (1, d_hidden)
    # offset = 0.15 * (1.0 - 0.0) = 0.15
    assert torch.allclose(offset, torch.ones(1, d_hidden) * 0.15)


def test_strategy_save_load(tmp_path):
    lib = StrategyLibrary()
    latent_state = torch.ones(1, 16)
    film_params = [(torch.ones(1, 16), torch.zeros(1, 16))]

    # Save strategy
    strat = lib.save_strategy("math_reasoning", latent_state, film_params, score=0.95)
    assert strat.task_type == "math_reasoning"
    assert strat.score == 0.95

    # Persist to disk and reload
    filepath = str(tmp_path / "strategies.pt")
    lib.save_to_disk(filepath)

    new_lib = StrategyLibrary()
    new_lib.load_from_disk(filepath)

    loaded_strats = new_lib.store.get_all("math_reasoning")
    assert len(loaded_strats) == 1
    assert loaded_strats[0].score == 0.95
    assert torch.allclose(loaded_strats[0].latent_state, latent_state)


def test_strategy_find_best():
    lib = StrategyLibrary(similarity_threshold=0.3)
    vec_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    vec_b = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    lib.save_strategy("code_gen", vec_a, score=1.0)
    lib.save_strategy("code_gen", vec_b, score=0.5)

    # Query with vec_a -> vec_a should rank top
    matches = lib.find_best("code_gen", vec_a, top_k=1)
    assert len(matches) == 1
    best_strat, score = matches[0]
    assert torch.allclose(best_strat.latent_state, vec_a)


def test_strategy_evolution():
    lib = StrategyLibrary()
    vec = torch.ones(1, 8)
    strat = lib.save_strategy("logic_puzzle", vec, score=0.8)

    # Mutate strategy
    mutated = lib.evolve_strategy(strat, noise_std=0.01)
    assert mutated.metadata.get("mutated") is True
    assert not torch.allclose(mutated.latent_state, vec)
    assert len(lib.store.get_all("logic_puzzle")) == 2


def test_breakthrough_engine_integration():
    d_model = 16

    # Mock components for CIDA Breakthrough Engine
    class MockConfig:
        d_hidden = 16
        num_agents = 4

    class MockCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockConfig()
        def forward(self, pooled):
            class FinalState:
                def __init__(self):
                    self.s = torch.ones(1, 4, 16)
            return {"state_final": FinalState()}

    class MockWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.cida_core = MockCore()
            self.d_llm = d_model
            self.param = nn.Parameter(torch.zeros(1))

    class MockTransformerLayer(nn.Module):
        def forward(self, x): return (x,)

    class MockLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([MockTransformerLayer(), MockTransformerLayer()])
            self.model.param = nn.Parameter(torch.zeros(1))
            self.model.generate = self.generate
            self.param = nn.Parameter(torch.zeros(1))


        def get_last_hidden(self, input_ids, attention_mask):
            return torch.ones(1, d_model)

        def generate(self, **kwargs):
            return torch.tensor([[10, 20, 30]])


    class MockTokenizer:
        def __call__(self, text, return_tensors="pt"):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]])
            }

        def decode(self, ids, skip_special_tokens=True):
            return "Breakthrough output: verified equation."

        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3, 4]

        @property
        def pad_token_id(self): return 0


    wrapper = MockWrapper()
    llm = MockLLM()
    tokenizer = MockTokenizer()

    engine = CIDABreakthroughEngine(
        wrapper=wrapper,
        llm=llm,
        tokenizer=tokenizer,
        use_film=True,
        num_film_layers=2
    )

    res = engine.solve(
        problem_prompt="Derive special relativity formula",
        domain_facts=["c = 3e8 m/s"],
        max_mcts_branches=2,
        max_new_tokens=64
    )

    assert "final_text" in res
    assert "total_time" in res
    assert len(res["trace_log"]) > 0
    assert res["best_score"] > -float('inf')
