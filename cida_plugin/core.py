"""
CIDAPlugin — Universal Multi-Agent Deliberation Layer (v3 Simplified).

Architecture:
    1. Input Projection: d_input → d_hidden
    2. Agent Initialization with Role Priors (structural disagreement)
    3. Deliberation Loop (evidence → messages → communication → update)
    4. Weighted Mean Consensus (correct for correlated agents)
    5. Uncertainty = f(disagreement) — observed, not learned
"""
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CIDAPluginConfig
from .agent import AgentState, apply_role_prior, compute_role_orthogonality_loss
from .deliberation import (
    AgentEvidenceExtractor,
    MessageFormulator,
    CounterargumentCommunication,
    AgentUpdater,
    PerspectiveProjector,
)
from .consensus import ConsensusAggregator, HaltingPredictor
from .ttt import TestTimeTrainer
from .liquid_dynamics import LiquidDeliberationSolver


class CIDAPlugin(nn.Module):
    """
    Universal CIDA-Plugin.

    Accepts pooled_output from ANY encoder via input_proj,
    runs multi-agent deliberation with structural role priors,
    returns calibrated class distribution with uncertainty.

    Example:
        cfg = CIDAPluginConfig(d_input=768, num_classes=2)
        plugin = CIDAPlugin(cfg)
        out = plugin(pooled)         # (B, 768) → dict
        logits = out["p_final"]      # (B, 2)
    """

    def __init__(self, config: CIDAPluginConfig):
        super().__init__()
        self.config = config
        cfg = config

        # ── Input projection: d_input → d_hidden ──────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.d_input, cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),
            nn.SiLU(),
        )
        if cfg.freeze_input_proj:
            for p in self.input_proj.parameters():
                p.requires_grad = False

        # Projection for sequence output (evidence pointers)
        self.seq_proj = (
            nn.Linear(cfg.d_input, cfg.d_hidden)
            if cfg.d_input != cfg.d_hidden
            else nn.Identity()
        )

        # ── Perspective Projector (optional per-agent projections) ─────────────
        if cfg.use_perspective_projector:
            self.perspective_projector = PerspectiveProjector(
                cfg.num_agents, cfg.d_input, cfg.d_hidden
            )
        else:
            self.perspective_projector = None

        # ── Deliberation components ───────────────────────────────────────────
        self.evidence_extractor = AgentEvidenceExtractor(cfg.d_hidden)
        self.message_formulator = MessageFormulator(
            cfg.d_hidden, cfg.num_classes, cfg.d_message,
            anonymous=cfg.anonymous_messages,
        )
        self.communication = CounterargumentCommunication(
            cfg.d_hidden, cfg.d_message
        )
        self.updater = AgentUpdater(
            cfg.d_hidden,
            cfg.num_classes,
            num_heads=cfg.num_attn_heads,
            multi_label=cfg.multi_label,
        )

        # ── Consensus (Weighted Mean for correlated agents) ───────────────────
        self.consensus = ConsensusAggregator(
            num_agents=cfg.num_agents,
            multi_label=cfg.multi_label,
            use_dynamic_trust=cfg.use_dynamic_trust,
            trust_ema_gamma=cfg.trust_ema_gamma,
        )
        self.halter = HaltingPredictor()

        # ── Gated communication warm-up ───────────────────────────────────────
        self.comm_gate = nn.Sequential(
            nn.Linear(cfg.d_hidden, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.comm_gate[0].bias, -0.5)

        # ── Test-Time Training (optional) ─────────────────────────────────────
        if cfg.use_ttt:
            self.ttt_trainer = TestTimeTrainer(
                d_hidden=cfg.d_hidden,
                ttt_steps=cfg.ttt_steps,
                ttt_lr=cfg.ttt_lr,
                mask_ratio=cfg.ttt_mask_ratio,
            )

        # ── Liquid Neural ODE Dynamics (optional) ─────────────────────────────
        if cfg.use_liquid_dynamics:
            self.liquid_solver = LiquidDeliberationSolver(
                extractor=self.evidence_extractor,
                message_formulator=self.message_formulator,
                communication=self.communication,
                updater=self.updater,
                d_hidden=cfg.d_hidden,
                solver=cfg.liquid_solver,
                atol=cfg.liquid_atol,
                rtol=cfg.liquid_rtol,
                T=float(cfg.max_rounds),
                trajectory_save_every=cfg.trajectory_save_every,
            )

    # ── Hugging Face Serialization ────────────────────────────────────────────

    def save_pretrained(self, save_directory: str):
        """Save config and weights for later loading or HF Hub publishing."""
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        weights_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        """Load a pretrained plugin (config + weights)."""
        config = CIDAPluginConfig.from_pretrained(
            pretrained_model_name_or_path
        )
        model = cls(config)

        if os.path.isdir(pretrained_model_name_or_path):
            weights_path = os.path.join(
                pretrained_model_name_or_path, "pytorch_model.bin"
            )
        else:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="pytorch_model.bin",
            )

        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        pooled: torch.Tensor,
        seq_output: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> dict:
        """
        Parameters
        ----------
        pooled     : (B, d_input)           pooled representation from encoder
        seq_output : (B, seq_len, d_input)  optional — for evidence pointers
        mask       : (B, seq_len)           padding mask

        Returns dict:
            p_final      : (B, K)     — final calibrated distribution
            b_all        : list       — belief history per round
            u_all        : list       — uncertainty history
            m_all        : list       — message history
            p_ptr_all    : list       — evidence pointer history
            uncertainty  : (B,)       — final uncertainty
            disagreement : (B,)       — final agent disagreement
            L_orth       : scalar     — orthogonality loss (representations)
            rounds_used  : int        — actual number of rounds used
            ponder_cost  : (B,) or None — NFE for liquid dynamics
            ttt_info     : dict or None — TTT diagnostics
        """
        cfg = self.config
        B = pooled.size(0)
        M = cfg.num_agents

        # ── Project input ─────────────────────────────────────────────────────
        h_cls = self.input_proj(pooled)  # (B, d_hidden)

        H = None
        if seq_output is not None:
            H = self.seq_proj(seq_output)  # (B, seq_len, d_hidden)

        # ── Initialize agents ─────────────────────────────────────────────────
        # All agents start from the same representation (no RoleEmbeddings).
        # Structural diversity comes from apply_role_prior on beliefs.
        if self.perspective_projector is not None:
            # Per-agent projections for diverse initial views
            s_t = self.perspective_projector(pooled)  # (B, M, d_hidden)
        else:
            # All agents start from the same representation
            s_t = h_cls.unsqueeze(1).expand(B, M, cfg.d_hidden).clone()

        seq_len = H.size(1) if H is not None else 1
        e_init = torch.zeros(B, M, cfg.d_hidden, device=pooled.device)
        r_init = torch.zeros_like(e_init)

        if cfg.multi_label:
            b_init = torch.full(
                (B, M, cfg.num_classes), 0.5, device=pooled.device
            )
        else:
            b_init = torch.zeros(
                B, M, cfg.num_classes, device=pooled.device
            )

        state_t = AgentState(
            s=s_t,
            b=b_init,
            u=torch.ones(B, M, 1, device=pooled.device),
            p=torch.zeros(B, M, seq_len, device=pooled.device),
            e=e_init,
            alpha=torch.ones(B, M, cfg.num_classes, device=pooled.device),
        )

        # Initial update to get first beliefs from data
        state_t = self.updater(state_t, r_init, e_init)

        # Apply structural role priors to initial beliefs
        beliefs = []
        for i in range(M):
            b_i = apply_role_prior(
                state_t.b[:, i, :].clone(),
                role_idx=i,
                multi_label=cfg.multi_label,
                positive_class_idx=cfg.positive_class_idx,
            )
            beliefs.append(b_i)
        state_t.b = torch.stack(beliefs, dim=1)

        # ── TTT: adapt before deliberation ────────────────────────────────────
        ttt_info = None
        _ttt_saved_updater = None
        _ttt_saved_head = None
        if cfg.use_ttt:
            ttt_info = self.ttt_trainer.adapt_and_get_info(
                self.updater, state_t.s
            )
            _ttt_saved_updater = ttt_info.pop("saved_updater_state")
            _ttt_saved_head = ttt_info.pop("saved_head_state")

        # ── Deliberation ──────────────────────────────────────────────────────
        b_all, u_all, m_all, p_ptr_all = [], [], [], []
        rounds_used = 0
        ponder_cost = None

        if cfg.use_liquid_dynamics and H is not None:
            # ── Continuous dynamics via ODE ────────────────────────────────────
            s_final, traj = self.liquid_solver(state_t.s, H, mask)
            b_all = traj["b_all"]
            u_all = traj["u_all"]
            p_ptr_all = traj["p_ptr_all"]
            m_all = traj["m_all"]

            if len(b_all) > 0:
                state_t.s = s_final
                # Final update to get beliefs from final state
                e_final = s_final
                if H is not None and not cfg.abl_no_pointers:
                    _, e_final = self.evidence_extractor(s_final, H, mask)

                r_final = torch.zeros_like(s_final)
                if not cfg.abl_no_messages and not cfg.abl_no_communication:
                    num_classes = self.updater.g[-1].out_features
                    dummy_b = torch.zeros(
                        B, M, num_classes, device=pooled.device
                    )
                    dummy_u = torch.ones(B, M, 1, device=pooled.device)
                    m_final = self.message_formulator(
                        s_final, e_final, dummy_b, dummy_u
                    )
                    r_final = self.communication(
                        s_final, m_final, dummy_b, e_final
                    )

                dummy_state = AgentState(
                    s=s_final,
                    b=b_all[-1] if b_all else state_t.b,
                    u=u_all[-1] if u_all else state_t.u,
                    p=torch.zeros(B, M, 1, device=pooled.device),
                    e=e_final,
                    alpha=torch.ones(
                        B, M, cfg.num_classes, device=pooled.device
                    ),
                )
                state_t = self.updater(dummy_state, r_final, e_final)

                # Apply role priors to final beliefs
                beliefs = []
                for i in range(M):
                    b_i = apply_role_prior(
                        state_t.b[:, i, :].clone(),
                        role_idx=i,
                        multi_label=cfg.multi_label,
                        positive_class_idx=cfg.positive_class_idx,
                    )
                    beliefs.append(b_i)
                state_t.b = torch.stack(beliefs, dim=1)
                b_all[-1] = state_t.b

            ponder_cost = torch.full(
                (B,), float(traj["nfe"]), device=pooled.device
            )
            rounds_used = traj["nfe"]

        else:
            # ── Discrete deliberation loop ────────────────────────────────────
            if cfg.use_liquid_dynamics and H is None:
                # ── Silent fallback warning ────────────────────────────────────
                warnings.warn(
                    "use_liquid_dynamics=True but seq_output is None. "
                    "Falling back to discrete deliberation mode. "
                    "Provide seq_output for continuous ODE dynamics.",
                    UserWarning,
                )

            for t in range(cfg.max_rounds):
                rounds_used = t + 1

                # Evidence Pointers
                if cfg.abl_no_pointers or H is None:
                    p = (
                        torch.ones(B, M, seq_len, device=pooled.device)
                        / seq_len
                    )
                    e_t = state_t.s
                else:
                    p, e_t = self.evidence_extractor(state_t.s, H, mask)

                state_t.p = p
                state_t.e = e_t
                p_ptr_all.append(p)

                # Message Formulation
                if cfg.abl_no_messages:
                    m_t = torch.zeros(
                        B, M, cfg.d_message, device=pooled.device
                    )
                else:
                    m_t = self.message_formulator(
                        state_t.s, state_t.e, state_t.b, state_t.u
                    )
                m_all.append(m_t)

                # Communication
                if cfg.abl_no_communication:
                    r_t = torch.zeros_like(state_t.s)
                else:
                    r_t = self.communication(
                        state_t.s, m_t, state_t.b, state_t.e,
                        sparse_k=cfg.sparse_comm_k,
                    )
                    gate = self.comm_gate(state_t.s)
                    r_t = gate * r_t
                    if self.training and cfg.comm_dropout > 0:
                        r_t = F.dropout(
                            r_t, p=cfg.comm_dropout, training=True
                        )

                # Agent Update
                state_t = self.updater(state_t, r_t, e_t)

                # Apply role priors to updated beliefs
                beliefs = []
                for i in range(M):
                    b_i = apply_role_prior(
                        state_t.b[:, i, :].clone(),
                        role_idx=i,
                        multi_label=cfg.multi_label,
                        positive_class_idx=cfg.positive_class_idx,
                    )
                    beliefs.append(b_i)
                state_t.b = torch.stack(beliefs, dim=1)

                b_all.append(state_t.b)
                u_all.append(state_t.u)

                # Confidence-based Early Stopping (inference only)
                if not self.training and cfg.early_stop_threshold is not None:
                    with torch.no_grad():
                        tmp_p, _, tmp_d = self.consensus(state_t.b)
                        if self.halter.should_halt(tmp_d).all():
                            break
                        if (
                            tmp_p.max(dim=-1).values.mean().item()
                            >= cfg.early_stop_threshold
                        ):
                            break

        # ── Final Consensus ───────────────────────────────────────────────────
        p_final, uncertainty, disagreement = self.consensus(
            state_t.b, state_t.u
        )

        # ── Orthogonality loss (on representations, not weights) ──────────────
        L_orth = compute_role_orthogonality_loss(state_t.s)

        # ── TTT: restore weights ──────────────────────────────────────────────
        if cfg.use_ttt and _ttt_saved_updater is not None:
            self.ttt_trainer.restore_weights(
                self.updater, _ttt_saved_updater, _ttt_saved_head
            )

        return {
            "p_final": p_final,
            "b_all": b_all,
            "u_all": u_all,
            "m_all": m_all,
            "p_ptr_all": p_ptr_all,
            "uncertainty": uncertainty,
            "disagreement": disagreement,
            "u_epi": state_t.u_epi,
            "u_alea": state_t.u_alea,
            "L_orth": L_orth,
            "state_final": state_t,
            "rounds_used": rounds_used,
            "ponder_cost": ponder_cost,
            "ttt_info": ttt_info,
        }
