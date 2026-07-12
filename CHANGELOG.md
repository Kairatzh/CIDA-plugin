# Changelog

## 1.4.0 — Latent Engineering & Multi-Vector Prompts

- **Latent Engineering Paradigm**: Introduced `LatentPrompt` (`cida_plugin.inference.latent_engineering`). A revolutionary feature allowing developers to compile heavy text prompts (e.g., 5000+ words) into multi-vector hidden state representations. Saves massive token costs, eliminates TTFT (Time To First Token) delay, and protects proprietary prompts.
- **Multi-Vector Chunking**: `LatentPrompt.compile_from_text()` splits giant instructions into distinct logical tensors, extracting a list of concepts. This prevents information bottlenecking and guarantees precise cross-attention matching during execution.
- **Latent Hub Integration**: `LatentPrompt.from_huggingface()` allows downloading vector prompts directly from Hugging Face Hub (e.g., `expert_coder.pt`) enabling an ecosystem of pre-compiled "Latent Brains".
- **Eternal Memory Injection**: `LatentAgent` now accepts `latent_prompt` at initialization, automatically injecting it into `LatentMemory` with infinite decay resistance (`decay_weight = 999.0`).
- **QA Checklist Generator**: Included `make_qa_notebook.py` to auto-generate `QA_Release_Checklist.ipynb` with 10 critical QA tests (Memory Leaks, Firewall, Dtype matching, Swarm Pipeline).

## 1.3.0 — CidaSwarm & Plan Cache Fix

- **`CidaSwarm`**: New multi-agent orchestrator class (`cida_plugin.inference.swarm`). Agents share a single `LatentMemory` blackboard. Three execution modes:
  - **Auto-route**: cosine-similarity routing between task embedding and agent role embeddings.
  - **Pipeline**: sequential agent execution (Agent A → Agent B → ...), each building on shared memory.
  - **Fan-out / Fan-in**: parallel agents + a merge agent that synthesizes all findings.
- **Plan Cache invalidation on tool error**: When a tool fails, stale cache entries (similarity > 0.90) are evicted. This prevents the "death loop" where the agent re-generates the same broken plan 3x within a single turn.
- **`SwarmCallbackHandler`**: New callback base class for swarm-level events (`on_route_decision`, `on_handoff`, `on_swarm_complete`).
- **Version bump**: `cida-plugin` → `1.3.0`.

## 1.2.0 — Agentic Inference Engine Integration

- **`cida_plugin.inference` is now canonical**: All agentic classes (`CIDAWrapper`, `LatentAgent`, `LatentMemory`, `LatentReasoner`) live inside the main package under `cida_plugin.inference`.
- **Backward-compatible `cida_inference` alias**: The standalone `cida_inference` package is now a shim that re-exports from `cida_plugin.inference`. All existing scripts (`run_cloud_operations_benchmark.py`, etc.) continue to work without any changes.
- **Latent Garbage Collector**: `LatentMemory` now tracks temporal `decay_weights` per stored vector. Unused memories decay by 10% per step; the top 2 retrieved memories are refreshed to 1.0. Prevents context pollution on long tasks.
- **Fast-Path Heuristics**: `LatentAgent` detects trivial tasks via embedding variance (threshold 0.05). If variance is below threshold, CIDA deliberation is bypassed entirely (`current_rounds = 0`), delivering LangChain-speed responses for simple queries while preserving full latent compute for hard problems.
- **Tool Exhaustion Circuit Breaker**: After 3 consecutive failures of the same tool, its identity vector is injected into the Latent Firewall, preventing the agent from calling it again. The broken tool is automatically unblocked on success.
- **Heuristic Observation Validators**: `register_tool(name, func, validator=None)` now accepts an optional Python callable. If the validator returns `False`, the raw observation is replaced with a standardized error string before the agent sees it — preventing hallucinations from corrupt API responses.
- **`requests` added to core dependencies**: Required for the built-in `call_web_api` tool.
- **Version bump**: `cida-plugin` → `1.2.0`.

## 3.0.0

- **Architecture Simplification**: Transitioned from 11-component Omega loss system to a 3-component `CIDALoss` (Task, Calibration, Anti-Collapse).
- **Bayesian Role Priors**: Replaced `RoleEmbeddings` and `debate_loss` with fixed structural agent priors (Prosecutor, Defender, Skeptic, Integrator) to mathematically guarantee agent disagreement.
- **Weighted Mean Consensus**: Replaced Product of Experts (PoE) with a correlation-aware weighted mean aggregation to solve overconfidence amplification.
- **Disagreement-as-Uncertainty**: Replaced the circular `ReliabilityTracker` with observable variance-based uncertainty quantification.
- **Liquid Dynamics & TTT Integration**: Unified continuous-time Neural ODE deliberation and Test-Time Training (TTT) with the simplified architecture.
- **Cleaned Up Technical Debt**: Deleted legacy inline-duplicated experiment scripts (`experiment.py`, `exp.py`) and temporary files.

## 2.0.0

- Replaced the earlier self-contrastive diversity loss with pairwise decorrelation
- Made round-weight sharing an explicit and configurable architectural choice
- Upgraded the deliberation residual to a gated opinion-revision update
- Replaced the loop-based sparse MoE path with a vectorized implementation
- Added packaging metadata, examples, tests, and GitHub-oriented project files
