# Changelog

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
