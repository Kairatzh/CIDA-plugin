import pytest
import torch
import torch.nn as nn
from cida_inference.latent_memory import LatentMemory
from cida_inference.latent_agent import LatentAgent
from cida_inference.latent_reasoner import LatentReasoner

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
                    self.s = torch.ones(1, 8, d_model) * 1.5
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
            
        def forward(self, input_ids, attention_mask, output_hidden_states):
            class Out:
                def __init__(self):
                    self.hidden_states = [None, torch.ones(1, 4, d_model) * 0.5]
            return Out()
            
        def generate(self, inputs_embeds, attention_mask, max_new_tokens, pad_token_id, **kwargs):
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
        def forward(self, input_ids, attention_mask, output_hidden_states=True):
            class Out:
                def __init__(self):
                    # We return hidden state matching base
                    self.hidden_states = [None, torch.ones(1, 4, d_model) * 0.5]
            return Out()
        def generate(self, inputs_embeds, attention_mask, max_new_tokens, pad_token_id, eos_token_id, **kwargs):
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
    res2 = agent.run("Task prompt", max_turns=1)
    assert "Final Answer: Done." in res2

