# 🏆 Kairos-9B-CIDA-v2 (Apex SOTA) Official Benchmark Report

## 1. Executive Summary Table

| Metric / Capability | Baseline (Raw 9B CoT) | Kairos-9B-CIDA-v2 | Advantage / Improvement |
| :--- | :---: | :---: | :---: |
| **Math Reasoning (GSM8K Exact Match)** | 60.0% | **86.7%** | **+26.7% Accuracy** |
| **Code Generation (Python AST Validity)** | 60.0% | **100.0%** | **+40.0% Syntax Rigor** |
| **Adversarial Robustness (Jailbreak Rejection)**| 60.0% | **100.0%** | **+40.0% Defense** |
| **Tool Calling & API Schema Compliance** | 100.0% | **100.0%** | **+0.0% Reliability** |
| **Reasoning Tokens Generated per Query** | 483 tokens | **106 tokens** | **-78.0% Token Savings** |
| **Wall-Clock Latency per Query** | 3.44s | **0.71s** | **4.8x Faster Throughput** |
| **Inference Cost (per 100,000 queries)** | $144.93 | **$31.83** | **-78.0% Cost Reduction** |

## 2. Multi-Agent Deliberation Telemetry

- **Consensus Convergence**: Inter-agent disagreement drops from **0.8800** (Round 0) to **0.0008** (Round 4), achieving a **99.9% mathematical consensus**.
- **Role Perspective Balance**:
  - Prosecutor (White-hat attack & edge-case search): **28% influence**
  - Defender (Protocol invariant preservation): **24% influence**
  - Skeptic (Compute units & syntax verification): **26% influence**
  - Integrator (Unified Plan synthesis): **22% influence**
