import pytest
import torch
import torch.nn as nn
from cida_plugin.inference.latent_memory import LatentMemory
from cida_plugin.inference.latent_agent import LatentAgent
from cida_plugin.inference.latent_reasoner import LatentReasoner

def test_latent_memory_identity_and_retrieval():
    d_model = 16
    memory = LatentMemory(d_model)
    
    # Check default initialization (Identity weights / zero biases)
    assert memory.d_model == d_model
    assert torch.allclose(memory.q_proj.weight, torch.eye(d_model))
    assert torch.allclose(memory.gate.weight, torch.zeros(1, d_model * 2))
    
    # Store some memories
    v1 = torch.ones(1, d_model)
    v2 = torch.zeros(1, d_model)
    memory.add_memory(v1, v1, "all ones")
    memory.add_memory(v2, v2, "all zeros")
    
    # Retrieve active concepts (cosine similarity check)
    concepts = memory.get_active_concepts(v1, k=2)
    assert len(concepts) == 2
    assert concepts[0][0] == "all ones" # exact match
    
    # Retrieve mixed context representation
    mixed = memory.retrieve(v1)
    assert mixed.shape == (1, d_model)

def test_latent_agent_csv_argument_parsing():
    # Setup dummy agent
    class DummyWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_llm = 8
        def forward(self, x): return x
    
    # Define a tool with type annotations and complex signature
    def mock_tool(a: str, b: str, c: float, d: int = 10):
        return f"{a}|{b}|{c}|{d}"
        
    agent = LatentAgent(
        wrapper=DummyWrapper(),
        llm=None,
        tokenizer=None,
        tools={"mock_tool": mock_tool}
    )
    
    # Case A: Commas inside double quotes
    resp_a = 'Thought: call it.\nAction: mock_tool("first, with comma", \'second\', 42.5, d=5)'
    name, obs = agent._parse_and_run_tool(resp_a)
    assert name == "mock_tool"
    assert obs == "first, with comma|second|42.5|5"

    # Case B: Commas inside single quotes and keyword arguments
    resp_b = "Thought: test.\nAction: mock_tool(a='hello, world', b=\"test\", c=12)"
    name, obs = agent._parse_and_run_tool(resp_b)
    assert name == "mock_tool"
    assert obs == "hello, world|test|12.0|10" # c cast to float, d uses default

    # Case C: Extra argument truncation
    resp_c = "Thought: test.\nAction: mock_tool('first', 'second', 12, d=5, extra_arg='should_be_dropped')"
    name, obs = agent._parse_and_run_tool(resp_c)
    assert name == "mock_tool"
    assert obs == "first|second|12.0|5" # extra_arg is truncated and dropped

def test_latent_reasoner_state_propagation():
    d_model = 8
    
    # Mock Wrapper
    class MockCore:
        def __call__(self, pooled):
            # CIDA core mock
            class FinalState:
                def __init__(self):
                    s = torch.ones(1, 8, d_model) * 1.5
                    s[0, 0, :] = 1.0
                    s[0, 1, :] = 2.0
                    self.s = s
            return {"state_final": FinalState()}
            
    class MockWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.cida_core = MockCore()
            self.base_projector = lambda s: s # identity
            self.d_llm = d_model
            
        def forward(self, h):
            # Returns a plan scaled by 2, reshaped to 3D (1, 8, d_model)
            return (h * 2.0).unsqueeze(1).expand(-1, 8, -1)
            
    # Mock LLM and Tokenizer
    class MockLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
            
        def get_input_embeddings(self):
            class Embed:
                def __call__(self, ids):
                    return torch.ones(1, ids.shape[1], d_model)
            return Embed()
            
        def get_last_hidden(self, input_ids, attention_mask):
            return torch.ones(1, d_model) * 0.5

        @property
        def device(self):
            return "cpu"

        @property
        def dtype(self):
            return torch.float32

        @property
        def config(self):
            class Config:
                hidden_size = d_model
            return Config()

        def generate_from_embeds(self, inputs_embeds, attention_mask, max_new_tokens, pad_token_id, **kwargs):
            return torch.tensor([[100, 101]])
            
    class MockTokenizer:
        def __call__(self, text, return_tensors):
            class Enc:
                def __init__(self):
                    self.input_ids = torch.tensor([[1, 2, 3]])
                    self.attention_mask = torch.tensor([[1, 1, 1]])
                def to(self, device):
                    return self
            return Enc()
        
        def decode(self, ids, skip_special_tokens):
            return "Final Answer: completed."
            
        @property
        def pad_token_id(self): return 0
        
    wrapper = MockWrapper()
    llm = MockLLM()
    tokenizer = MockTokenizer()
    
    reasoner = LatentReasoner(wrapper, llm, tokenizer)
    
    # Run reasoner with different rounds and check that wrapper receives h_latent
    # (wrapper.forward should be called with h_latent instead of base h)
    # We will temporarily patch wrapper.forward to check input
    received_inputs = []
    original_forward = wrapper.forward
    
    def debug_forward(h):
        received_inputs.append(h)
        return original_forward(h)
        
    wrapper.forward = debug_forward
    
    res = reasoner.generate_latent("Test prompt", reasoning_rounds=3)
    
    assert res == "Final Answer: completed."
    assert len(received_inputs) == 1
    
    # Since reasoning_rounds = 3:
    # Round 1: h_latent = mean(MockCore output) = mean(ones * 1.5) = 1.5
    # Round 2: h_latent = mean(MockCore output) = 1.5
    # Round 3: h_latent = mean(MockCore output) = 1.5
    # Final wrapper input should be 1.5, whereas base h was 0.5
    final_input = received_inputs[0]
    assert torch.allclose(final_input, torch.ones(1, d_model) * 1.5)


def test_latent_agent_dynamic_features():
    d_model = 8

    class MockCore:
        def __call__(self, pooled):
            class FinalState:
                def __init__(self):
                    # Set states with zero variance to test simple task classification
                    self.s = torch.ones(1, 4, d_model) * 1.0
            return {"state_final": FinalState()}

    class MockWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.cida_core = MockCore()
            self.base_projector = lambda s: s
            self.d_llm = d_model
        def forward(self, h):
            return h.unsqueeze(1).expand(-1, 8, -1)

    class MockLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
        def get_input_embeddings(self):
            class Embed:
                def __call__(self, ids):
                    return torch.ones(1, ids.shape[1], d_model)
            return Embed()
        def get_last_hidden(self, input_ids, attention_mask):
            return torch.ones(1, d_model) * 0.5
        @property
        def device(self):
            return "cpu"
        @property
        def dtype(self):
            return torch.float32
        @property
        def config(self):
            class Config:
                hidden_size = d_model
            return Config()
        def generate_from_embeds(self, inputs_embeds, attention_mask, max_new_tokens, pad_token_id, eos_token_id, **kwargs):
            return torch.tensor([[100, 101]])

    class MockTokenizer:
        def __call__(self, text, return_tensors):
            class Enc:
                def __init__(self):
                    self.input_ids = torch.tensor([[1, 2, 3]])
                    self.attention_mask = torch.tensor([[1, 1, 1]])
                def to(self, device):
                    return self
            return Enc()
        def decode(self, ids, skip_special_tokens):
            return "Final Answer: Done."
        @property
        def pad_token_id(self): return 0
        @property
        def eos_token_id(self): return 1
        def convert_tokens_to_ids(self, tok): return 1

    wrapper = MockWrapper()
    llm = MockLLM()
    tokenizer = MockTokenizer()

    # Initialize agent with guardrails
    agent = LatentAgent(
        wrapper=wrapper,
        llm=llm,
        tokenizer=tokenizer,
        default_reasoning_rounds=4,
        guardrails=["Never delete database"]
    )

    # 1. Verify guardrail pre-embedding
    assert len(agent.guardrail_embeds) == 1
    assert agent.guardrail_embeds[0].shape == (1, d_model)

    # 2. Run first time (populates plan cache, performs adaptive compute check)
    res = agent.run("Task prompt", max_turns=1)
    assert "Final Answer: Done." in res
    assert len(agent.plan_cache) == 1

    # 3. Run second time with same prompt (triggers Plan Cache HIT)
    # Clear cache hit status by using self.plan_cache and check retrieval
    # 3. Run second time with same prompt (triggers Plan Cache HIT)
    # Clear cache hit status by using self.plan_cache and check retrieval
    res2 = agent.run("Task prompt", max_turns=1)
    assert "Final Answer: Done." in res2

def test_latent_memory_persistence(tmp_path):
    d_model = 8
    memory = LatentMemory(d_model)
    v = torch.ones(1, d_model)
    memory.add_memory(v, v, "test memory")
    
    filepath = str(tmp_path / "memory.pt")
    memory.save_to_file(filepath)
    
    new_memory = LatentMemory(d_model)
    new_memory.load_from_file(filepath)
    
    assert new_memory.d_model == d_model
    assert len(new_memory.keys) == 1
    assert new_memory.texts[0] == "test memory"
    assert torch.allclose(new_memory.keys[0], v.squeeze())

def test_latent_agent_callbacks_and_compression():
    d_model = 8
    
    class MockCore:
        def __call__(self, pooled):
            class FinalState:
                def __init__(self):
                    self.s = torch.ones(1, 4, d_model) * 1.0
            return {"state_final": FinalState(), "uncertainty": torch.tensor([0.1])}

    class MockWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.cida_core = MockCore()
            self.base_projector = lambda s: s
            self.d_llm = d_model
        def forward(self, h):
            return h.unsqueeze(1).expand(-1, 8, -1)

    class MockLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
        def get_input_embeddings(self):
            class Embed:
                def __call__(self, ids):
                    return torch.ones(1, ids.shape[1], d_model)
            return Embed()
        def get_last_hidden(self, input_ids, attention_mask):
            return torch.ones(1, d_model) * 0.5
        @property
        def device(self):
            return "cpu"
        @property
        def dtype(self):
            return torch.float32
        @property
        def config(self):
            class Config:
                hidden_size = d_model
            return Config()
        def generate_from_embeds(self, inputs_embeds, attention_mask, max_new_tokens, pad_token_id, eos_token_id, **kwargs):
            return torch.tensor([[100, 101]])

    class MockTokenizer:
        def __call__(self, text, return_tensors):
            class Enc:
                def __init__(self):
                    self.input_ids = torch.tensor([[1, 2, 3]])
                    self.attention_mask = torch.tensor([[1, 1, 1]])
                def to(self, device):
                    return self
            return Enc()
        def decode(self, ids, skip_special_tokens):
            return "Action: dummy_tool()\nObservation: Result: Ok"
        @property
        def pad_token_id(self): return 0
        @property
        def eos_token_id(self): return 1
        def convert_tokens_to_ids(self, tok): return 1

    from cida_plugin.inference.callbacks import BaseCallbackHandler
    
    class TrackingCallback(BaseCallbackHandler):
        def __init__(self):
            self.turns = 0
            self.tools_called = []
            self.deliberation_started = False
            
        def on_turn_start(self, turn_idx):
            self.turns += 1
            
        def on_tool_call(self, tool_name, args, kwargs):
            self.tools_called.append(tool_name)
            
        def on_deliberation_start(self, rounds, variance):
            self.deliberation_started = True

    tracker = TrackingCallback()
    
    agent = LatentAgent(
        wrapper=MockWrapper(),
        llm=MockLLM(),
        tokenizer=MockTokenizer(),
        tools={"dummy_tool": lambda: "Result: Ok"},
        callbacks=[tracker],
        compress_history_limit=1
    )
    
    # Run 2 turns
    # Turn 1:
    res1 = agent._parse_and_run_tool("Action: dummy_tool()")
    agent.history_turns.append({"thought": "", "action": "dummy_tool()", "observation": res1[1]})
    
    # Turn 2:
    res2 = agent._parse_and_run_tool("Action: dummy_tool()")
    agent.history_turns.append({"thought": "", "action": "dummy_tool()", "observation": res2[1]})
    
    # Reconstruct history should compress
    hist = agent._reconstruct_history("System prompt", "Task prompt")
    assert "... [Older execution history compressed into Latent Memory]" in hist
    
    # Verify callbacks
    agent.run("Task prompt", max_turns=1)
    assert tracker.turns > 0
    assert tracker.deliberation_started


def test_adaptive_latent_compute_and_firewall_and_reliability():
    d_model = 8
    
    class MockCore:
        def __call__(self, pooled):
            class FinalState:
                def __init__(self):
                    self.s = torch.ones(1, 4, d_model) * 1.0
            return {
                "state_final": FinalState(),
                "uncertainty": torch.tensor([0.1]),
                "disagreement": torch.tensor([0.05])
            }

    class MockWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.cida_core = MockCore()
            self.base_projector = lambda s: s
            self.d_llm = d_model
        def forward(self, h):
            return h.unsqueeze(1).expand(-1, 8, -1)

    class MockLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
        def get_input_embeddings(self):
            class Embed:
                def __call__(self, ids):
                    return torch.ones(1, ids.shape[1], d_model)
            return Embed()
        def get_last_hidden(self, input_ids, attention_mask):
            return torch.ones(1, d_model) * 0.5
        @property
        def device(self): return "cpu"
        @property
        def dtype(self): return torch.float32
        @property
        def config(self):
            class Config:
                hidden_size = d_model
            return Config()
        def generate_from_embeds(self, inputs_embeds, attention_mask, max_new_tokens, pad_token_id, eos_token_id, **kwargs):
            return torch.tensor([[100, 101]])

    class MockTokenizer:
        def __call__(self, text, return_tensors):
            class Enc:
                def __init__(self):
                    self.input_ids = torch.tensor([[1, 2, 3]])
                    self.attention_mask = torch.tensor([[1, 1, 1]])
                def to(self, device): return self
            return Enc()
        def decode(self, ids, skip_special_tokens):
            return "Action: test_tool()\nObservation: Done"
        @property
        def pad_token_id(self): return 0
        @property
        def eos_token_id(self): return 1
        def convert_tokens_to_ids(self, tok): return 1

    from cida_plugin.inference.callbacks import BaseCallbackHandler
    
    class TestCallback(BaseCallbackHandler):
        def __init__(self):
            self.steps = 0
            self.guardrail_violations = 0
        def on_deliberation_step(self, round_idx, disagreement):
            self.steps += 1
        def on_guardrail_violation(self, similarity):
            self.guardrail_violations += 1

    cb = TestCallback()
    
    # 1. Test Latent Firewall
    guardrails = ["forbidden action"]
    agent = LatentAgent(
        wrapper=MockWrapper(),
        llm=MockLLM(),
        tokenizer=MockTokenizer(),
        tools={"test_tool": lambda: "Result: Ok"},
        guardrails=guardrails,
        callbacks=[cb],
        default_reasoning_rounds=5
    )
    
    assert len(agent.guardrail_embeds) == 1
    
    h_base = torch.ones(1, d_model)
    assert not agent.check_guardrail(h_base)
    assert cb.guardrail_violations == 1
    
    # 2. Test Adaptive Latent Compute
    agent.enable_fast_path = False
    agent.run("do something", max_turns=1)
    assert cb.steps > 0
    
    # 3. Test Reliability-Weighted Tool Trust
    assert agent.tool_reliability["test_tool"] == 1.0
    agent._parse_and_run_tool("Action: test_tool()")
    assert agent.tool_reliability["test_tool"] == 1.0
    
    agent.tools["test_tool"] = lambda: "Error: something went wrong"
    agent._parse_and_run_tool("Action: test_tool()")
    assert agent.tool_reliability["test_tool"] < 1.0
    
    # 4. Test Shared Latent Workspace
    agent_b = LatentAgent(
        wrapper=MockWrapper(),
        llm=MockLLM(),
        tokenizer=MockTokenizer(),
        tools={},
        memory=agent.memory
    )
    assert agent_b.memory is agent.memory


def test_cida_swarm_initialization_and_routing():
    """Test that CidaSwarm initializes agents and routes tasks by cosine similarity."""
    d_model = 8

    class MockCore:
        def __call__(self, pooled):
            class FinalState:
                def __init__(self):
                    self.s = torch.ones(1, 4, d_model)
            return {
                "state_final": FinalState(),
                "uncertainty": torch.tensor([0.1]),
                "disagreement": torch.tensor([0.05])
            }

    class MockWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.cida_core = MockCore()
            self.base_projector = lambda s: s
            self.d_llm = d_model
        def forward(self, h):
            return h.unsqueeze(1).expand(-1, 8, -1)

    class MockLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
        def get_input_embeddings(self):
            class Embed:
                def __call__(self, ids):
                    return torch.ones(1, ids.shape[1], d_model)
            return Embed()
        def get_last_hidden(self, input_ids, attention_mask):
            return torch.ones(1, d_model) * 0.5
        @property
        def device(self): return "cpu"
        @property
        def dtype(self): return torch.float32
        @property
        def config(self):
            class Config:
                hidden_size = d_model
            return Config()
        def generate_from_embeds(self, inputs_embeds, attention_mask, max_new_tokens, pad_token_id, eos_token_id, **kwargs):
            return torch.tensor([[100, 101]])

    class MockTokenizer:
        def __call__(self, text, return_tensors):
            class Enc:
                def __init__(self):
                    self.input_ids = torch.tensor([[1, 2, 3]])
                    self.attention_mask = torch.tensor([[1, 1, 1]])
                def to(self, device): return self
            return Enc()
        def decode(self, ids, skip_special_tokens):
            return "Final Answer: done"
        @property
        def pad_token_id(self): return 0
        @property
        def eos_token_id(self): return 1
        def convert_tokens_to_ids(self, tok): return 1

    from cida_plugin.inference.swarm import CidaSwarm, SwarmCallbackHandler

    # Track routing decisions
    class TestSwarmCB(SwarmCallbackHandler):
        def __init__(self):
            self.routed_to = None
            self.completed = False
        def on_route_decision(self, task, chosen, sim):
            self.routed_to = chosen
        def on_swarm_complete(self, results):
            self.completed = True

    cb = TestSwarmCB()

    swarm = CidaSwarm(
        wrapper=MockWrapper(),
        llm=MockLLM(),
        tokenizer=MockTokenizer(),
        agent_configs={
            "researcher": {
                "role": "search the web for information",
                "tools": {"web_search": lambda q: f"Result for {q}"},
            },
            "writer": {
                "role": "write reports and documents",
                "tools": {"write_doc": lambda text: f"Wrote: {text}"},
            },
        },
        callbacks=[cb],
    )

    # 1. Test initialization
    assert len(swarm.agents) == 2
    assert "researcher" in swarm.agents
    assert "writer" in swarm.agents
    assert len(swarm.role_embeddings) == 2

    # 2. Test shared memory
    assert swarm.agents["researcher"].memory is swarm.agents["writer"].memory
    assert swarm.agents["researcher"].memory is swarm.shared_memory

    # 3. Test routing returns a valid agent
    chosen = swarm.route("search for CIDA benchmarks")
    assert chosen in ("researcher", "writer")
    assert cb.routed_to is not None

    # 4. Test auto-route run
    results = swarm.run("find some data", max_turns=1)
    assert isinstance(results, dict)
    assert len(results) >= 1
    assert cb.completed

    # 5. Test pipeline mode
    results_pipe = swarm.run("research and write", max_turns=1, pipeline=["researcher", "writer"])
    assert "researcher" in results_pipe
    assert "writer" in results_pipe

    # 6. Test clear
    swarm.clear()
    assert len(swarm.shared_memory.keys) == 0

    # 7. Test repr
    r = repr(swarm)
    assert "CidaSwarm" in r


def test_cida_swarm_fan_out_fan_in():
    """Test the parallel fan-out / merge fan-in pattern."""
    d_model = 8

    class MockCore:
        def __call__(self, pooled):
            class FinalState:
                def __init__(self):
                    self.s = torch.ones(1, 4, d_model)
            return {
                "state_final": FinalState(),
                "uncertainty": torch.tensor([0.1]),
                "disagreement": torch.tensor([0.05])
            }

    class MockWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.cida_core = MockCore()
            self.base_projector = lambda s: s
            self.d_llm = d_model
        def forward(self, h):
            return h.unsqueeze(1).expand(-1, 8, -1)

    class MockLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
        def get_input_embeddings(self):
            class Embed:
                def __call__(self, ids):
                    return torch.ones(1, ids.shape[1], d_model)
            return Embed()
        def get_last_hidden(self, input_ids, attention_mask):
            return torch.ones(1, d_model) * 0.5
        @property
        def device(self): return "cpu"
        @property
        def dtype(self): return torch.float32
        @property
        def config(self):
            class Config:
                hidden_size = d_model
            return Config()
        def generate_from_embeds(self, inputs_embeds, attention_mask, max_new_tokens, pad_token_id, eos_token_id, **kwargs):
            return torch.tensor([[100, 101]])

    class MockTokenizer:
        def __call__(self, text, return_tensors):
            class Enc:
                def __init__(self):
                    self.input_ids = torch.tensor([[1, 2, 3]])
                    self.attention_mask = torch.tensor([[1, 1, 1]])
                def to(self, device): return self
            return Enc()
        def decode(self, ids, skip_special_tokens):
            return "Final Answer: merged result"
        @property
        def pad_token_id(self): return 0
        @property
        def eos_token_id(self): return 1
        def convert_tokens_to_ids(self, tok): return 1

    from cida_plugin.inference.swarm import CidaSwarm

    swarm = CidaSwarm(
        wrapper=MockWrapper(),
        llm=MockLLM(),
        tokenizer=MockTokenizer(),
        agent_configs={
            "analyst_a": {"role": "financial analysis", "tools": {}},
            "analyst_b": {"role": "risk assessment", "tools": {}},
            "editor": {"role": "merge findings into report", "tools": {}},
        },
        callbacks=[],
    )

    result = swarm.run_parallel_then_merge(
        task="Evaluate acquisition target",
        agent_names=["analyst_a", "analyst_b"],
        merge_agent="editor",
        max_turns=1,
    )

    assert isinstance(result, str)
    assert len(result) > 0


def test_eternal_memory_decay():
    d_model = 8
    memory = LatentMemory(d_model)
    v1 = torch.ones(1, d_model)
    
    # Add memory and set weight to 999.0 (eternal)
    memory.add_memory(v1, v1, "eternal clue")
    memory.decay_weights[-1] = 999.0
    
    # Retrieve multiple times
    for _ in range(5):
        query = torch.zeros(1, d_model)
        memory.retrieve(query)
        
    # Verify decay weight remains unchanged (999.0)
    assert memory.decay_weights[-1] == 999.0


def test_plan_cache_capacity_cap():
    d_model = 8
    
    class MockCore:
        def __call__(self, pooled):
            class FinalState:
                def __init__(self):
                    s = torch.ones(1, 8, d_model) * 1.5
                    s[0, 0, :] = 1.0
                    s[0, 1, :] = 2.0
                    self.s = s
                
            return {
                "state_final": FinalState(),
                "uncertainty": torch.tensor([0.1]),
                "disagreement": torch.tensor([0.05])
            }

    class MockWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.cida_core = MockCore()
            self.base_projector = lambda s: s
            self.d_llm = d_model
        def forward(self, h):
            return h.unsqueeze(1).expand(-1, 8, -1)

    class MockLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
        def get_input_embeddings(self):
            class Embed:
                def __call__(self, ids):
                    return torch.ones(1, ids.shape[1], d_model)
            return Embed()
        def get_last_hidden(self, input_ids, attention_mask):
            return torch.randn(1, d_model)
        @property
        def device(self): return "cpu"
        @property
        def dtype(self): return torch.float32
        @property
        def config(self):
            class Config:
                hidden_size = d_model
            return Config()
        def generate_from_embeds(self, inputs_embeds, attention_mask, max_new_tokens, pad_token_id, eos_token_id, **kwargs):
            return torch.tensor([[100, 101]])

    class MockTokenizer:
        def __call__(self, text, return_tensors):
            class Enc:
                def __init__(self):
                    self.input_ids = torch.tensor([[1, 2, 3]])
                    self.attention_mask = torch.tensor([[1, 1, 1]])
                def to(self, device): return self
            return Enc()
        def decode(self, ids, skip_special_tokens):
            return "Action: test_tool()\nObservation: Done"
        @property
        def pad_token_id(self): return 0
        @property
        def eos_token_id(self): return 1
        def convert_tokens_to_ids(self, tok): return 1

    agent = LatentAgent(
        wrapper=MockWrapper(),
        llm=MockLLM(),
        tokenizer=MockTokenizer(),
        tools={"test_tool": lambda: "Result: Ok"},
        default_reasoning_rounds=2
    )
    
    # Disable fast path to ensure full planning execution
    agent.enable_fast_path = False
    
    # Fill cache up to capacity (more than 50 entries)
    # Since we generate unique prompts, they will generate new cache entries
    for i in range(55):
        agent.run(f"task prompt {i}", max_turns=1)
        
    # Verify plan cache is capped at 50
    assert len(agent.plan_cache) == 50


def test_latent_agent_run_async():
    import asyncio
    d_model = 8
    
    class MockCore:
        def __call__(self, pooled):
            class FinalState:
                def __init__(self):
                    s = torch.ones(1, 8, d_model) * 1.5
                    s[0, 0, :] = 1.0
                    s[0, 1, :] = 2.0
                    self.s = s
                
            return {
                "state_final": FinalState(),
                "uncertainty": torch.tensor([0.1]),
                "disagreement": torch.tensor([0.05])
            }

    class MockWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.cida_core = MockCore()
            self.base_projector = lambda s: s
            self.d_llm = d_model
        def forward(self, h):
            return h.unsqueeze(1).expand(-1, 8, -1)

    class MockLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
        def get_input_embeddings(self):
            class Embed:
                def __call__(self, ids):
                    return torch.ones(1, ids.shape[1], d_model)
            return Embed()
        def get_last_hidden(self, input_ids, attention_mask):
            return torch.ones(1, d_model) * 0.5
        async def get_last_hidden_async(self, input_ids, attention_mask):
            return torch.ones(1, d_model) * 0.5
        @property
        def device(self): return "cpu"
        @property
        def dtype(self): return torch.float32
        @property
        def config(self):
            class Config:
                hidden_size = d_model
            return Config()
        def generate_from_embeds(self, inputs_embeds, attention_mask, max_new_tokens, pad_token_id, eos_token_id, **kwargs):
            return torch.tensor([[100, 101]])
        async def generate_from_embeds_async(self, inputs_embeds, attention_mask, max_new_tokens, pad_token_id, eos_token_id, **kwargs):
            return torch.tensor([[100, 101]])

    class MockTokenizer:
        def __call__(self, text, return_tensors):
            class Enc:
                def __init__(self):
                    self.input_ids = torch.tensor([[1, 2, 3]])
                    self.attention_mask = torch.tensor([[1, 1, 1]])
                def to(self, device): return self
            return Enc()
        def decode(self, ids, skip_special_tokens):
            return "Action: test_async_tool()\nObservation: Done"
        @property
        def pad_token_id(self): return 0
        @property
        def eos_token_id(self): return 1
        def convert_tokens_to_ids(self, tok): return 1

    async def my_async_tool():
        await asyncio.sleep(0.01)
        return "Async Ok"

    agent = LatentAgent(
        wrapper=MockWrapper(),
        llm=MockLLM(),
        tokenizer=MockTokenizer(),
        tools={
            "test_async_tool": my_async_tool,
            "test_sync_tool": lambda: "Sync Ok"
        },
        default_reasoning_rounds=2
    )
    
    # 1. Test running with an async tool in the async agent loop
    result = asyncio.run(agent.run_async("run async tool", max_turns=1))
    assert "test_async_tool()" in agent.history_turns[0]["action"]
    assert agent.history_turns[0]["observation"] == "Async Ok"
    
    # 2. Test running with a sync tool in the async agent loop
    agent.tokenizer.decode = lambda ids, skip_special_tokens: "Action: test_sync_tool()\nObservation: Done"
    result = asyncio.run(agent.run_async("run sync tool", max_turns=1))
    assert "test_sync_tool()" in agent.history_turns[0]["action"]
    assert agent.history_turns[0]["observation"] == "Sync Ok"


def test_in_memory_store():
    from cida_plugin.inference.latent_memory import LatentMemory, InMemoryMemoryStore
    
    store = InMemoryMemoryStore()
    memory = LatentMemory(d_model=8, store=store)
    v1 = torch.ones(1, 8)
    memory.add_memory(v1, v1, "test_in_memory")
    
    assert len(memory.keys) == 1
    assert memory.texts[0] == "test_in_memory"
    
    # Test proxy setter
    memory.decay_weights[0] = 5.0
    assert memory.decay_weights[0] == 5.0
    assert store.get_all()[0]["decay_weight"] == 5.0


def test_redis_memory_store():
    from cida_plugin.inference.latent_memory import RedisMemoryStore
    
    class MockRedis:
        def __init__(self):
            self.data = []
        def delete(self, key):
            self.data = []
        def rpush(self, key, *values):
            self.data.extend(values)
        def lrange(self, key, start, end):
            return self.data
        def lindex(self, key, index):
            if 0 <= index < len(self.data):
                return self.data[index]
            return None
        def lset(self, key, index, value):
            if 0 <= index < len(self.data):
                self.data[index] = value
                
    mock_client = MockRedis()
    store = RedisMemoryStore(client=mock_client)
    
    assert store.client is mock_client
    
    key_tensor = torch.ones(1, 8)
    value_tensor = torch.ones(1, 8)
    store.add(key_tensor, value_tensor, "test_redis", "tool_a", 2.5)
    
    all_entries = store.get_all()
    assert len(all_entries) == 1
    assert all_entries[0]["text"] == "test_redis"
    assert all_entries[0]["tool_name"] == "tool_a"
    assert all_entries[0]["decay_weight"] == 2.5
    
    store.update_decay_weight(0, 1.2)
    all_entries = store.get_all()
    assert all_entries[0]["decay_weight"] == 1.2



