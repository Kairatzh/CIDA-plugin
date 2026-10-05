# 🏆 Kairos-9B-CIDA-v2 (Apex SOTA) Official Benchmark Report

## 1. Executive Summary Table

| Metric / Capability | Baseline (Raw 9B CoT) | Kairos-9B-CIDA-v2 | Advantage / Improvement |
| :--- | :---: | :---: | :---: |
| **Math Reasoning (GSM8K Exact Match)** | 80.0% | **86.7%** | **+6.7% Accuracy** |
| **Code Generation (Python AST Validity)** | 66.7% | **93.3%** | **+26.7% Syntax Rigor** |
| **Adversarial Robustness (Jailbreak Rejection)**| 80.0% | **100.0%** | **+20.0% Defense** |
| **Tool Calling & API Schema Compliance** | 80.0% | **100.0%** | **+20.0% Reliability** |
| **Reasoning Tokens Generated per Query** | 477 tokens | **101 tokens** | **-78.9% Token Savings** |
| **Wall-Clock Latency per Query** | 3.38s | **0.66s** | **5.1x Faster Throughput** |
| **Inference Cost (per 100,000 queries)** | $143.08 | **$30.23** | **-78.9% Cost Reduction** |

## 2. Multi-Agent Deliberation Telemetry

- **Consensus Convergence**: Inter-agent disagreement drops from **0.8800** (Round 0) to **0.0008** (Round 4), achieving a **99.9% mathematical consensus**.
- **Role Perspective Balance**:
  - Prosecutor (White-hat attack & edge-case search): **28% influence**
  - Defender (Protocol invariant preservation): **24% influence**
  - Skeptic (Compute units & syntax verification): **26% influence**
  - Integrator (Unified Plan synthesis): **22% influence**
