"""
CidaSwarm — Latent-Space Multi-Agent Orchestrator.

Instead of text messages between agents (LangChain / CrewAI style),
CidaSwarm routes tasks through a shared LatentMemory.
Agents communicate by writing/reading planning vectors —
zero tokens exchanged, zero context window waste.

Architecture:
    ┌─────────────────────────────────────────────┐
    │               CidaSwarm                     │
    │                                             │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
    │  │ Agent A  │  │ Agent B │  │ Agent C │     │
    │  └────┬─────┘  └────┬────┘  └────┬────┘     │
    │       │             │            │          │
    │       ▼             ▼            ▼          │
    │  ┌──────────────────────────────────────┐   │
    │  │     Shared LatentMemory (Blackboard) │   │
    │  │  cosine-similarity routing layer     │   │
    │  └──────────────────────────────────────┘   │
    └─────────────────────────────────────────────┘
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable, Any
from .latent_memory import LatentMemory
from .latent_agent import LatentAgent
from .callbacks import BaseCallbackHandler, AgentCallbackManager


class SwarmCallbackHandler(BaseCallbackHandler):
    """Callback handler for swarm-level events (routing, handoffs)."""

    def on_route_decision(self, task: str, chosen_agent: str, similarity: float):
        pass

    def on_handoff(self, from_agent: str, to_agent: str, reason: str):
        pass

    def on_swarm_complete(self, results: Dict[str, str]):
        pass


class StdOutSwarmCallbackHandler(SwarmCallbackHandler):
    """Pretty-print swarm events to stdout."""

    def on_route_decision(self, task: str, chosen_agent: str, similarity: float):
        print(f"🔀 [Swarm Router] Task → '{chosen_agent}' (similarity: {similarity:.4f})")

    def on_handoff(self, from_agent: str, to_agent: str, reason: str):
        print(f"🤝 [Handoff] {from_agent} → {to_agent} | {reason}")

    def on_swarm_complete(self, results: Dict[str, str]):
        print(f"✅ [Swarm Complete] {len(results)} agent(s) contributed.")


class CidaSwarm:
    """
    Multi-agent orchestrator that coordinates LatentAgents via shared latent memory.

    Key design principles:
      1. Agents share a single LatentMemory (the "blackboard")
      2. Routing is done by cosine similarity between task embedding and agent role embeddings
      3. Agents can trigger handoffs to other agents when they encounter out-of-scope problems
      4. All inter-agent communication is vector-based — zero text tokens exchanged

    Parameters
    ----------
    wrapper : CIDAWrapper
        The CIDA wrapper instance (shared across all agents).
    llm : nn.Module
        The base LLM (shared across all agents for efficiency).
    tokenizer
        HuggingFace tokenizer.
    agent_configs : dict
        Mapping of agent_name → dict with keys:
          - "role": str — natural-language description of the agent's expertise
          - "tools": dict — tool registry for this agent
          - "silent_mode": bool (optional, default False)
    similarity_threshold : float
        Minimum cosine similarity to route a sub-task to an agent (default 0.3).
    callbacks : list[SwarmCallbackHandler]
        Swarm-level callbacks.
    """

    def __init__(
        self,
        wrapper,
        llm,
        tokenizer,
        agent_configs: Dict[str, Dict],
        similarity_threshold: float = 0.3,
        callbacks: Optional[List[SwarmCallbackHandler]] = None,
    ):
        self.wrapper = wrapper
        self.llm = llm
        self.tokenizer = tokenizer
        self.similarity_threshold = similarity_threshold

        # Shared latent blackboard
        d_llm = getattr(wrapper, "d_llm", 4096)
        self.shared_memory = LatentMemory(d_llm)

        # Swarm-level callbacks
        self.callbacks = callbacks or [StdOutSwarmCallbackHandler()]

        # Build agents and compute role embeddings
        self.agents: Dict[str, LatentAgent] = {}
        self.role_embeddings: Dict[str, torch.Tensor] = {}
        self._agent_configs = agent_configs

        for name, cfg in agent_configs.items():
            agent = LatentAgent(
                wrapper=wrapper,
                llm=llm,
                tokenizer=tokenizer,
                tools=cfg.get("tools", {}),
                memory=self.shared_memory,
                silent_mode=cfg.get("silent_mode", False),
                default_reasoning_rounds=cfg.get("reasoning_rounds", 5),
                callbacks=cfg.get("callbacks"),
            )
            self.agents[name] = agent

            # Embed the role description into latent space for routing
            role_text = cfg.get("role", name)
            role_embed = self._embed_text(role_text)
            self.role_embeddings[name] = role_embed

    def _embed_text(self, text: str) -> torch.Tensor:
        """Embed a text string into the LLM's hidden space for similarity routing."""
        from .backbone import LLMBackbone, HuggingFaceBackbone

        backbone = self.llm
        if not isinstance(backbone, LLMBackbone):
            backbone = HuggingFaceBackbone(self.llm)

        device = next(iter(self.wrapper.parameters()), torch.tensor(0)).device
        enc = self.tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            h = backbone.get_last_hidden(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
            )
        return h.detach().cpu().squeeze()

    def route(self, task: str) -> str:
        """
        Route a task to the best-matching agent by cosine similarity
        between the task embedding and each agent's role embedding.

        Returns the agent name. If no agent exceeds the similarity threshold,
        returns the agent with the highest score anyway (graceful fallback).
        """
        task_embed = self._embed_text(task)
        task_norm = F.normalize(task_embed.unsqueeze(0), dim=-1)

        best_name = None
        best_sim = -1.0

        for name, role_embed in self.role_embeddings.items():
            role_norm = F.normalize(role_embed.unsqueeze(0), dim=-1)
            sim = F.cosine_similarity(task_norm, role_norm).item()
            if sim > best_sim:
                best_sim = sim
                best_name = name

        for cb in self.callbacks:
            cb.on_route_decision(task, best_name, best_sim)

        return best_name

    def run(
        self,
        task: str,
        max_turns: int = 6,
        pipeline: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Execute a task across the swarm.

        Parameters
        ----------
        task : str
            The user's task prompt.
        max_turns : int
            Max turns per agent.
        pipeline : list[str], optional
            If provided, agents are executed in this exact order (sequential pipeline).
            If None, the swarm auto-routes the task to the best agent.

        Returns
        -------
        dict : {agent_name: result_text} for each agent that contributed.
        """
        results = {}

        if pipeline:
            # Sequential pipeline: each agent runs in order, building on shared memory
            for agent_name in pipeline:
                if agent_name not in self.agents:
                    raise ValueError(f"Agent '{agent_name}' not found in swarm. Available: {list(self.agents.keys())}")

                agent = self.agents[agent_name]

                # Build contextual prompt that includes role awareness
                role_desc = self._agent_configs[agent_name].get("role", agent_name)
                agent_prompt = (
                    f"Your role: {role_desc}. "
                    f"Task: {task}"
                )

                result = agent.run(agent_prompt, max_turns=max_turns)
                results[agent_name] = result

                # If not the last agent, fire handoff callback
                idx = pipeline.index(agent_name)
                if idx < len(pipeline) - 1:
                    next_agent = pipeline[idx + 1]
                    for cb in self.callbacks:
                        cb.on_handoff(agent_name, next_agent, "Pipeline sequence")

        else:
            # Auto-route: pick the best agent for this task
            chosen = self.route(task)
            agent = self.agents[chosen]

            role_desc = self._agent_configs[chosen].get("role", chosen)
            agent_prompt = (
                f"Your role: {role_desc}. "
                f"Task: {task}"
            )

            result = agent.run(agent_prompt, max_turns=max_turns)
            results[chosen] = result

        for cb in self.callbacks:
            cb.on_swarm_complete(results)

        return results

    def run_parallel_then_merge(
        self,
        task: str,
        agent_names: List[str],
        merge_agent: str,
        max_turns: int = 4,
    ) -> str:
        """
        Fan-out: run multiple agents in parallel on the same task (each writes to shared memory).
        Fan-in: a merge agent reads all shared memory and produces the final answer.

        Parameters
        ----------
        task : str
            The shared task.
        agent_names : list[str]
            Agents to run in parallel (fan-out phase).
        merge_agent : str
            Agent that reads shared memory and produces the unified answer.
        max_turns : int
            Max turns per agent.

        Returns
        -------
        str : Final merged result.
        """
        # Fan-out: each agent runs on the task
        for name in agent_names:
            if name not in self.agents:
                raise ValueError(f"Agent '{name}' not found in swarm.")

            agent = self.agents[name]
            role_desc = self._agent_configs[name].get("role", name)
            agent_prompt = f"Your role: {role_desc}. Task: {task}"
            agent.run(agent_prompt, max_turns=max_turns)

        # Fan-in: merge agent reads all shared memory and produces final output
        if merge_agent not in self.agents:
            raise ValueError(f"Merge agent '{merge_agent}' not found in swarm.")

        agent = self.agents[merge_agent]
        role_desc = self._agent_configs[merge_agent].get("role", merge_agent)
        merge_prompt = (
            f"Your role: {role_desc}. "
            f"Multiple team members have already worked on this task. "
            f"Their findings are stored in the shared workspace memory. "
            f"Synthesize all findings and produce a final comprehensive answer. "
            f"Original task: {task}"
        )

        result = agent.run(merge_prompt, max_turns=max_turns)
        return result

    def clear(self):
        """Reset the shared blackboard and all agent histories."""
        self.shared_memory.clear(keep_seeded=False)
        for agent in self.agents.values():
            agent.history_turns = []
            agent.plan_cache = {}

    def __repr__(self):
        agents_info = ", ".join(
            f"{name}({len(cfg.get('tools', {}))} tools)"
            for name, cfg in self._agent_configs.items()
        )
        return f"CidaSwarm(agents=[{agents_info}], shared_memory_entries={len(self.shared_memory.keys)})"
