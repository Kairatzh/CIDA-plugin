# 🖥️ CIDA SRE Demo: 72 Hours Autonomous Infrastructure Management

This folder contains a complete simulation of a 40+ microservices fintech/e-commerce environment managed by a `CidaSwarm` of agents.

## 🚀 Key Features Demonstrated
1. **Silent Deliberation:** The agent reasoner conducts agent-consensus debates in the latent space before any output tokens are generated.
2. **Infinite Context:** Using `LatentMemory` with cosine retrieval allows agents to retain key historical facts and anomalies for 72+ hours without overflowing the LLM prompt context window.
3. **Agentic Resilience:** Real-time error trapping, rollback, and automatic tool replacement in response to API errors and cascading failures.

## 📂 Modules
- `simulator.py`: High-fidelity stateful mock of Kubernetes clusters, databases, Kafka, and networks.
- `scenarios.py`: Configurable definition of the 20 test cases and incidents.
- `agent_swarm.py`: Orchestration logic for the CIDA SRE Swarm.
- `run_demo.py`: Command line tool to run automated tests.
- `debug_runner.ipynb`: Interactive Jupyter dashboard with debugger and metrics plots.
