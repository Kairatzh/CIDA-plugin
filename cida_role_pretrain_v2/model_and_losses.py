"""
RoleCIDA v2 — Model wrapper + Losses + Training engine.
All fixes applied: connected graph, clean losses, EMA, gradient verification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from collections import defaultdict

from role_identity import (
    RoleIdentityLayer,
    RoleConditionedMessageFormulator,
    ContrastiveMessageLoss,
    RoleTimeDynamics,
)


class RoleCIDA(nn.Module):
    """
    v2: Role anchors are INJECTED into CIDAPlugin's initial agent states.
    
    Data flow:
        x (B, d_emb) 
          → RoleIdentityLayer → s_init (B, M, d_hidden)
          → CIDAPlugin uses s_init as starting agent states
          → Role-Conditioned Messages at each round
          → Consensus
    """
    def __init__(self, d_emb, d_hidden, num_classes, plugin_cls, plugin_cfg,
                 embed_fn=None):
        super().__init__()
        self.d_emb = d_emb
        self.d_hidden = d_hidden
        self.num_classes = num_classes

        # Role identity layer (produces s_init for agents)
        self.role_layer = RoleIdentityLayer(
            d_emb=d_emb, d_hidden=d_hidden, n_agents=4, embed_fn=embed_fn
        )
        # CIDA plugin (we disable its internal PerspectiveProjector)
        plugin_cfg.use_perspective_projector = False
        self.plugin = plugin_cls(plugin_cfg)
        # Override plugin's message formulator with role-conditioned version
        self.role_msg = RoleConditionedMessageFormulator(
            d_hidden=d_hidden,
            num_classes=num_classes,
            d_message=plugin_cfg.d_message,
            n_agents=4,
            anonymous=plugin_cfg.anonymous_messages,
        )
        # Contrastive message loss
        self.msg_contrast = ContrastiveMessageLoss(
            d_message=plugin_cfg.d_message, n_agents=4
        )
        # Role time dynamics
        self.time_dynamics = RoleTimeDynamics(n_agents=4, d_hidden=d_hidden)

    def forward(self, x):
        """
        x: (B, d_emb) — pre-computed embedding
        """
        # 1) Role-conditioned initial states (FIX: anchors now IN the graph)
        s_init = self.role_layer(x)  # (B, M, d_hidden)

        # 2) Inject s_init into plugin by monkey-patching perspective projector
        #    We pass raw x to plugin but override the initial state
        out = self._forward_with_role_init(x, s_init)

        # 3) Add role-specific outputs for loss computation
        out['anchor_div_loss'] = self.role_layer.anchor_diversity_loss()
        out['msg_contrast_loss'] = self.msg_contrast(out.get('m_all', []))
        out['tau'] = self.time_dynamics(x[:, :self.d_hidden]
                                         if x.size(-1) >= self.d_hidden
                                         else self.role_layer.input_proj(x))
        out['anchor_sim'] = self.role_layer.anchor_similarity_matrix()
        return out

    def _forward_with_role_init(self, x, s_init):
        """Run CIDAPlugin but replace agent initialization with s_init."""
        from cida_plugin.agent import AgentState, apply_role_priors_batched
        cfg = self.plugin.config
        B = x.size(0)
        M = cfg.num_agents
        K = cfg.num_classes

        # Project input (for evidence extraction)
        h_cls = self.plugin.input_proj(x)  # (B, d_hidden)

        # Use role-conditioned s_init instead of PerspectiveProjector
        s_t = s_init  # (B, M, d_hidden) — FROM RoleIdentityLayer

        seq_len = 1
        e_init = torch.zeros(B, M, cfg.d_hidden, device=x.device)
        r_init = torch.zeros_like(e_init)

        if cfg.multi_label:
            b_init = torch.full((B, M, K), 0.5, device=x.device)
        else:
            b_init = torch.zeros(B, M, K, device=x.device)

        state_t = AgentState(
            s=s_t, b=b_init,
            u=torch.ones(B, M, 1, device=x.device),
            p=torch.zeros(B, M, seq_len, device=x.device),
            e=e_init,
            alpha=torch.ones(B, M, K, device=x.device),
        )

        # Initial update
        state_t = self.plugin.updater(state_t, r_init, e_init)
        state_t.b = apply_role_priors_batched(
            state_t.b, M, multi_label=cfg.multi_label,
            positive_class_idx=cfg.positive_class_idx,
        )

        # Deliberation loop with role-conditioned messages
        b_all, u_all, m_all = [], [], []
        for t in range(cfg.max_rounds):
            # Evidence (no seq_output → use state as evidence)
            p_new = torch.ones(B, M, seq_len, device=x.device) / seq_len
            e_new = state_t.s
            state_t.p = p_new
            state_t.e = e_new

            # Role-conditioned messages (v2 improvement)
            m_new = self.role_msg(state_t.s, state_t.e, state_t.b, state_t.u)

            # Communication
            r_new = self.plugin.communication(
                state_t.s, m_new, state_t.b, state_t.e,
                sparse_k=cfg.sparse_comm_k
            )
            gate = self.plugin.comm_gate(state_t.s)
            r_new = gate * r_new
            if self.training and cfg.comm_dropout > 0:
                r_new = F.dropout(r_new, p=cfg.comm_dropout, training=True)

            # Update
            state_t = self.plugin.updater(state_t, r_new, e_new)
            state_t.b = apply_role_priors_batched(
                state_t.b, M, multi_label=cfg.multi_label,
                positive_class_idx=cfg.positive_class_idx,
            )

            b_all.append(state_t.b)
            u_all.append(state_t.u)
            m_all.append(m_new)

        # Consensus
        p_final, uncertainty, disagreement = self.plugin.consensus(
            state_t.b, state_t.u
        )
        from cida_plugin.agent import compute_role_orthogonality_loss
        L_orth = compute_role_orthogonality_loss(state_t.s)

        return {
            'p_final': p_final,
            'b_all': b_all,
            'u_all': u_all,
            'm_all': m_all,
            'uncertainty': uncertainty,
            'disagreement': disagreement,
            'L_orth': L_orth,
            'rounds_used': cfg.max_rounds,
            'state_final': state_t,
        }


class RolePreTrainLossV2(nn.Module):
    """
    v2: Behavioral loss with fixes:
    - Anchor diversity on RAW anchors (not blended)
    - Consensus coherence properly scaled
    - Added contrastive message loss integration
    """
    def __init__(self, n_classes=3):
        super().__init__()
        self.K = n_classes

    def forward(self, b_all, arg_roles, anchor_div_loss, msg_contrast_loss,
                stage, anneal=1.0):
        if not b_all:
            return torch.tensor(0.0, requires_grad=True), {}

        eps = 1e-7
        B = arg_roles.size(0)
        K = self.K
        device = arg_roles.device

        pro_mask = (arg_roles == 0)
        unc_mask = (arg_roles == 1)
        con_mask = (arg_roles == 2)

        total_loss = torch.tensor(0.0, device=device)
        info = defaultdict(float)
        max_ent = math.log(K)

        round_w = torch.linspace(0.4, 1.0, len(b_all), device=device)
        round_w = round_w / round_w.sum()

        for rnd, (b_t, w_r) in enumerate(zip(b_all, round_w)):
            _, M, _ = b_t.shape
            b_t = b_t[:, :, :K].clamp(eps, 1-eps)
            b_t = b_t / b_t.sum(-1, keepdim=True)

            agent_losses = []
            for i in range(min(M, 4)):
                b_i = b_t[:, i, :]
                conf = b_i.max(dim=1).values
                ent = -(b_i * b_i.log()).sum(dim=1)
                b_pro = b_i[:, 0]
                b_con = b_i[:, min(2, K-1)]

                if i == 0:  # PROSECUTOR
                    l_pro = -torch.log(b_pro[pro_mask] + eps).mean() if pro_mask.any() else torch.tensor(0., device=device)
                    l_con = torch.clamp(b_con[con_mask] - 0.5, min=0).mean() if con_mask.any() else torch.tensor(0., device=device)
                    l_unc = -torch.log(b_pro[unc_mask] + eps).mean() * 0.3 if unc_mask.any() else torch.tensor(0., device=device)
                    loss_i = l_pro + 0.3*l_con + 0.2*l_unc
                    info['prosecutor'] += loss_i.item()
                elif i == 1:  # DEFENDER
                    l_con = -torch.log(b_con[con_mask] + eps).mean() if con_mask.any() else torch.tensor(0., device=device)
                    l_pro = torch.clamp(b_pro[pro_mask] - 0.5, min=0).mean() if pro_mask.any() else torch.tensor(0., device=device)
                    l_unc = -torch.log(b_con[unc_mask] + eps).mean() * 0.3 if unc_mask.any() else torch.tensor(0., device=device)
                    loss_i = l_con + 0.3*l_pro + 0.2*l_unc
                    info['defender'] += loss_i.item()
                elif i == 2:  # SKEPTIC
                    l_unc = -ent[unc_mask].mean() / max_ent if unc_mask.any() else torch.tensor(0., device=device)
                    certain_mask = ~unc_mask
                    l_cert = torch.clamp(conf[certain_mask] - 0.7, min=0).mean() if certain_mask.any() else torch.tensor(0., device=device)
                    loss_i = l_unc + 0.4*l_cert
                    info['skeptic'] += loss_i.item()
                else:  # INTEGRATOR
                    y_c = arg_roles.clamp(0, K-1)
                    loss_i = F.nll_loss(b_i.log(), y_c)
                    info['integrator'] += loss_i.item()

                agent_losses.append(loss_i)

            # Consensus coherence
            b_mean = b_t.mean(dim=1, keepdim=True)
            integrator_b = b_t[:, 3:4, :]
            coherence = F.kl_div(
                (integrator_b + eps).log(), b_mean,  # FIX: no eps on target
                reduction='batchmean'
            )
            info['coherence'] += coherence.item()

            round_total = (torch.stack(agent_losses).mean()
                          + 0.05 * coherence * anneal)
            total_loss = total_loss + w_r * round_total

        n_rounds = len(b_all)
        # FIX: anchor diversity on RAW anchors (passed in, not computed from blended)
        total_loss = total_loss + 0.1 * anchor_div_loss
        info['anchor_div'] = anchor_div_loss.item()

        # NEW: contrastive message loss
        total_loss = total_loss + 0.05 * msg_contrast_loss * anneal
        info['msg_contrast'] = msg_contrast_loss.item()

        return total_loss, {k: v/n_rounds if k not in ('anchor_div','msg_contrast') else v
                           for k, v in info.items()}


class UncertaintyCalibrationLossV2(nn.Module):
    """v2: Fixed KL target normalization."""
    def forward(self, b_all, arg_roles, soft_labels=None):
        if not b_all:
            return torch.tensor(0.0, requires_grad=True), {}
        eps = 1e-7
        K = b_all[0].shape[-1]

        if soft_labels is None:
            y_oh = F.one_hot(arg_roles.clamp(0, K-1), K).float()
            alpha = 0.7
            soft = alpha * y_oh + (1-alpha)/K
            # FIX: ensure soft sums to 1.0 (no eps contamination)
            soft = soft / soft.sum(dim=-1, keepdim=True)
        else:
            soft = soft_labels

        b_final = b_all[-1][:, :, :K].clamp(eps, 1-eps)
        b_final = b_final / b_final.sum(-1, keepdim=True)

        losses = []
        for i in range(b_final.size(1)):
            b_i = b_final[:, i, :]
            if i == 2:  # Skeptic
                kl = F.kl_div((b_i+eps).log(), soft, reduction='batchmean')
                losses.append(kl)
            elif i == 3:  # Integrator
                kl = F.kl_div((b_i+eps).log(), soft, reduction='batchmean')
                losses.append(kl * 0.8)
            else:
                ce = F.nll_loss(b_i.log(), arg_roles.clamp(0, K-1))
                losses.append(ce * 0.5)

        loss = torch.stack(losses).mean()
        return loss, {'calib': loss.item()}


class EMAModel:
    """Exponential Moving Average of model weights for stable evaluation."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model):
        """Temporarily apply EMA weights."""
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        """Restore original weights."""
        model.load_state_dict(self._backup)
        del self._backup
