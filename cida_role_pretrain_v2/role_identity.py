"""
RoleIdentityLayer v2 — Fixed: anchors now feed INTO CIDAPlugin.
+ Role-Conditioned Messaging
+ Contrastive Message Loss (InfoNCE)
+ Role-specific time constants τ
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict


class RoleIdentityLayer(nn.Module):
    """
    Learnable role anchors injected DIRECTLY into agent init states.
    
    v2 fixes:
    - Anchors are projected into d_hidden (not d_emb) so they match CIDAPlugin
    - forward() returns (B, M, d_hidden) ready for plugin.s_init
    - anchor_diversity_loss() computed on RAW anchors, not blended vectors
    """
    ROLE_PROTOTYPES = {
        0: [  # Prosecutor
            'The evidence clearly demonstrates this is true.',
            'Multiple studies confirm this conclusion beyond doubt.',
            'This is definitively supported by the data.',
            'The facts unambiguously point to this being correct.',
            'There is overwhelming support for this position.',
        ],
        1: [  # Defender
            'However, this argument fails to account for key exceptions.',
            'The opposing evidence suggests we should reconsider.',
            'This claim is not supported when examined carefully.',
            'There are significant flaws in this line of reasoning.',
            'Alternative explanations are equally or more plausible.',
        ],
        2: [  # Skeptic
            'It is unclear whether this conclusion is warranted.',
            'The evidence is mixed and difficult to interpret.',
            'We cannot be certain given the available information.',
            'This remains an open question with no clear answer.',
            'Further investigation is needed before drawing conclusions.',
        ],
        3: [  # Integrator
            'Taking all perspectives into account, the most balanced view is.',
            'Weighing the evidence from all sides leads to this synthesis.',
            'Both arguments have merit and the truth likely lies between them.',
            'Integrating these viewpoints gives us a more complete picture.',
            'The consensus emerging from all perspectives suggests.',
        ],
    }

    def __init__(self, d_emb, d_hidden, n_agents=4, embed_fn=None):
        super().__init__()
        self.n_agents = n_agents
        self.d_hidden = d_hidden

        # Initialize anchors from prototypes if embed_fn provided
        if embed_fn is not None:
            anchors = []
            for i in range(n_agents):
                proto_vecs = embed_fn(self.ROLE_PROTOTYPES[i])
                anchor = proto_vecs.mean(0)
                anchors.append(anchor)
            self.anchors = nn.Parameter(torch.stack(anchors))  # (M, d_emb)
        else:
            self.anchors = nn.Parameter(torch.randn(n_agents, d_emb) * 0.02)

        # Project anchors from d_emb -> d_hidden (matches CIDAPlugin)
        self.anchor_proj = nn.Sequential(
            nn.Linear(d_emb, d_hidden),
            nn.LayerNorm(d_hidden),
        )
        # Input projection (same as CIDAPlugin.input_proj)
        self.input_proj = nn.Sequential(
            nn.Linear(d_emb, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
        )
        # Learnable blend factor per agent
        self.blend_alpha = nn.Parameter(torch.full((n_agents,), 0.3))

    def forward(self, x):
        """
        x: (B, d_emb) — raw embedding from encoder
        Returns: (B, M, d_hidden) — role-conditioned initial states for agents
        """
        B = x.size(0)
        h = self.input_proj(x)  # (B, d_hidden)
        h_exp = h.unsqueeze(1).expand(B, self.n_agents, -1)  # (B, M, d_hidden)

        anchors_norm = F.normalize(self.anchors, p=2, dim=-1)
        anchors_h = self.anchor_proj(anchors_norm)  # (M, d_hidden)
        anchors_exp = anchors_h.unsqueeze(0).expand(B, -1, -1)  # (B, M, d_hidden)

        alpha = self.blend_alpha.sigmoid().view(1, self.n_agents, 1)  # (1, M, 1)
        combined = (1 - alpha) * h_exp + alpha * anchors_exp
        return combined  # (B, M, d_hidden)

    def get_anchor_embeds(self):
        """Return projected anchors for use in losses (detached from input)."""
        with torch.no_grad():
            a = F.normalize(self.anchors, p=2, dim=-1)
        return self.anchor_proj(a)  # (M, d_hidden)

    def anchor_diversity_loss(self):
        """Orthogonality on RAW anchors (fix: no input data contamination)."""
        a = F.normalize(self.anchors, p=2, dim=-1)  # (M, d_emb)
        gram = a @ a.T
        eye = torch.eye(self.n_agents, device=a.device)
        return ((gram - eye) ** 2).mean()

    def anchor_similarity_matrix(self):
        a = F.normalize(self.anchors, p=2, dim=-1)
        return (a @ a.T).detach().cpu()


class RoleConditionedMessageFormulator(nn.Module):
    """
    v2: Role anchor is concatenated to agent state BEFORE message generation.
    This forces each agent to produce role-characteristic messages at every round.
    """
    def __init__(self, d_hidden, num_classes, d_message, n_agents=4, anonymous=True):
        super().__init__()
        self.anonymous = anonymous
        self.n_agents = n_agents
        # Role embedding added to message input
        self.role_embed = nn.Embedding(n_agents, d_hidden)
        if anonymous:
            in_dim = d_hidden + num_classes + 1 + d_hidden  # [e; b; u; role]
        else:
            in_dim = d_hidden * 2 + num_classes + 1 + d_hidden  # [s; e; b; u; role]
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_message),
            nn.LayerNorm(d_message),
            nn.SiLU(),
            nn.Linear(d_message, d_message),
        )

    def forward(self, s, e, b, u):
        B, M, _ = s.shape
        role_ids = torch.arange(M, device=s.device).unsqueeze(0).expand(B, -1)
        role_emb = self.role_embed(role_ids)  # (B, M, d_hidden)
        if self.anonymous:
            concat = torch.cat([e, b, u, role_emb], dim=-1)
        else:
            concat = torch.cat([s, e, b, u, role_emb], dim=-1)
        return self.proj(concat)


class ContrastiveMessageLoss(nn.Module):
    """
    InfoNCE loss: a classifier should identify WHO sent a message
    from its latent representation. Forces agents to develop unique
    communication signatures.
    """
    def __init__(self, d_message, n_agents=4, temperature=0.07):
        super().__init__()
        self.classifier = nn.Linear(d_message, n_agents)
        self.temperature = temperature

    def forward(self, m_all):
        """
        m_all: list of (B, M, d_message) per round
        Returns: scalar loss
        """
        if not m_all:
            return torch.tensor(0.0, requires_grad=True)
        losses = []
        for m_t in m_all:
            B, M, d = m_t.shape
            m_flat = m_t.reshape(B * M, d)  # (B*M, d)
            labels = torch.arange(M, device=m_t.device).unsqueeze(0).expand(B, -1).reshape(-1)
            logits = self.classifier(m_flat) / self.temperature
            losses.append(F.cross_entropy(logits, labels))
        return torch.stack(losses).mean()


class RoleTimeDynamics(nn.Module):
    """
    Role-specific time constants τ for deliberation dynamics.
    
    Skeptic: high τ (resistant to persuasion, "viscous")
    Integrator: low τ (rapidly adapts, "fluid")
    Prosecutor/Defender: medium τ
    """
    def __init__(self, n_agents=4, d_hidden=256):
        super().__init__()
        # Learnable base τ per agent, initialized by role
        init_tau = torch.tensor([1.0, 1.0, 2.0, 0.5])  # P, D, S, I
        self.log_tau = nn.Parameter(torch.log(init_tau[:n_agents]))
        # Data-dependent modulation
        self.tau_mod = nn.Sequential(
            nn.Linear(d_hidden, n_agents),
            nn.Sigmoid(),
        )

    def forward(self, s_mean):
        """
        s_mean: (B, d_hidden) — mean agent state
        Returns: (B, M) — per-agent time constants
        """
        base_tau = self.log_tau.exp().unsqueeze(0)  # (1, M)
        mod = self.tau_mod(s_mean)  # (B, M) in [0, 1]
        return base_tau * (0.5 + mod)  # τ ∈ [0.5·base, 1.5·base]
