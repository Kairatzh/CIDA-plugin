"""
CIDAPlugin v2 — Universal Evidence-Grounded Multi-Agent Deliberation Layer.

Что нового в v2:
  1. TransformerAgentUpdater: GRU заменён на cross-attention + gated evidence fusion.
  2. EMA Reliability Tracker: rho обновляется автоматически на основе согласованности
     агентов с консенсусом. Больше не нужно передавать rho снаружи.
  3. Learnable D-Schedule: расписание разногласий обучается внутри OmegaLossSystem.
  4. Все параметры управляются через CIDAPluginConfig.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CIDAPluginConfig
from .agent import AgentState, RoleEmbeddings
from .deliberation import (
    AgentEvidenceExtractor,
    MessageFormulator,
    CounterargumentCommunication,
    AgentUpdater,
)
from .consensus import ConsensusAggregator, HaltingPredictor


class CIDAPlugin(nn.Module):
    """
    Универсальный CIDA-Plugin v2.

    Принимает pooled_output от ЛЮБОГО энкодера через input_proj,
    запускает многоагентный deliberation, возвращает калиброванное
    распределение над классами.

    Пример:
        cfg = CIDAPluginConfig(d_input=768, num_classes=2)
        plugin = CIDAPlugin(cfg)
        out = plugin(pooled)         # (B, 768) → dict
        logits = out["p_final"]      # (B, 2)
    """

    def __init__(self, config: CIDAPluginConfig):
        super().__init__()
        self.config = config
        cfg = config

        # ── Входной проекционный слой ─────────────────────────────────────────
        # Переводит любую размерность d_input → d_hidden.
        # Это единственная часть, которая "знает" про внешний энкодер.
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.d_input, cfg.d_hidden),
            nn.LayerNorm(cfg.d_hidden),
            nn.SiLU(),
        )
        if cfg.freeze_input_proj:
            for p in self.input_proj.parameters():
                p.requires_grad = False

        # Проекция seq_output для evidence pointers
        self.seq_proj = (
            nn.Linear(cfg.d_input, cfg.d_hidden)
            if cfg.d_input != cfg.d_hidden
            else nn.Identity()
        )

        # ── Компоненты deliberation ───────────────────────────────────────────
        self.role_embeddings = RoleEmbeddings(cfg.num_agents, cfg.d_hidden)
        self.evidence_extractor = AgentEvidenceExtractor(cfg.d_hidden)
        self.message_formulator = MessageFormulator(
            cfg.d_hidden, cfg.num_classes, cfg.d_message
        )
        self.communication = CounterargumentCommunication(cfg.d_hidden, cfg.d_message)

        # [v2] TransformerAgentUpdater с num_attn_heads из конфига
        self.updater = AgentUpdater(
            cfg.d_hidden, cfg.num_classes, num_heads=cfg.num_attn_heads
        )

        self.consensus = ConsensusAggregator()
        self.halter = HaltingPredictor(cfg.d_hidden, cfg.num_agents)

        # Gated communication warm-up
        self.comm_gate = nn.Sequential(
            nn.Linear(cfg.d_hidden, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.comm_gate[0].bias, -2.0)

        # ── [v2] EMA Reliability Tracker ─────────────────────────────────────
        # rho_ema хранит надёжность каждого агента как скользящее среднее.
        # Обновляется автоматически на основе согласия с консенсусом.
        # register_buffer: сохраняется в state_dict, но НЕ обучается через grad.
        self.register_buffer(
            "rho_ema",
            torch.ones(1, cfg.num_agents, 1)  # начинаем с единицы (равные агенты)
        )

    # ── Hugging Face Serialization ────────────────────────────────────────────

    def save_pretrained(self, save_directory: str):
        """
        Сохраняет конфигурацию и веса плагина для последующей загрузки
        или публикации на Hugging Face Hub.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        
        weights_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        """
        Загружает предобученный плагин (конфиг + веса).
        Поддерживает локальные пути и загрузку с Hugging Face Hub.
        """
        config = CIDAPluginConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        
        if os.path.isdir(pretrained_model_name_or_path):
            weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        else:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, 
                filename="pytorch_model.bin"
            )
            
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    # ── EMA Reliability Update ────────────────────────────────────────────────

    @torch.no_grad()
    def _update_rho(self, b_all: list, p_final: torch.Tensor):
        """
        Обновляет EMA надёжности агентов после каждого forward pass.

        Логика: агент надёжен, если его финальное верование близко к
        групповому консенсусу (p_final). Надёжность = 1 - L1(b_i, p_final).

        Обновление происходит только во время обучения (self.training).
        """
        b_final = b_all[-1]                                    # (B, M, K)
        p_expanded = p_final.unsqueeze(1).expand_as(b_final)  # (B, M, K)

        # agreement ∈ [0, 1]: 1 = полное согласие с консенсусом
        agreement = 1.0 - torch.norm(b_final - p_expanded, p=1, dim=-1, keepdim=True)
        agreement = agreement.clamp(min=0.0)   # (B, M, 1)

        # EMA по батчу: берём среднее по примерам
        batch_rho = agreement.mean(dim=0, keepdim=True)  # (1, M, 1)

        # Экспоненциальное скользящее среднее
        decay = self.config.rho_decay
        self.rho_ema = decay * self.rho_ema + (1.0 - decay) * batch_rho

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        pooled: torch.Tensor,
        seq_output: torch.Tensor = None,
        mask: torch.Tensor = None,
        rho: torch.Tensor = None,
        epoch_fraction: float = 0.0,
    ) -> dict:
        """
        Параметры
        ----------
        pooled      : (B, d_input)              pooled representation от энкодера
        seq_output  : (B, seq_len, d_input)     опционально — для evidence pointers
        mask        : (B, seq_len)              маска паддинга
        rho         : (B, M, 1) or None        переопределить EMA rho (для тестов)
        epoch_fraction : float                 прогресс обучения [0, 1]

        Возвращает dict:
            p_final     : (B, K)      — финальное калиброванное распределение
            b_all       : list        — история верований агентов
            u_all       : list        — история неопределённостей
            m_all       : list        — история сообщений
            p_ptr_all   : list        — история evidence pointers
            h_all       : list        — история сигналов halting
            L_dom       : scalar      — штраф за доминирование
            L_role      : scalar      — штраф за несовместимость ролей
            state_final : AgentState  — финальное состояние
            w_final     : (B, M)      — веса агентов в консенсусе
            rounds_used : int         — фактическое число раундов
            rho_snapshot: (M,)        — текущие EMA надёжности агентов
        """
        cfg = self.config
        B = pooled.size(0)
        M = cfg.num_agents

        # ── Проекция входа ────────────────────────────────────────────────────
        h_cls = self.input_proj(pooled)   # (B, d_input) → (B, d_hidden)

        H = None
        if seq_output is not None:
            H = self.seq_proj(seq_output)  # (B, seq_len, d_input) → (B, seq_len, d_hidden)

        # ── Инициализация агентов ─────────────────────────────────────────────
        roles = self.role_embeddings()   # (M, d_hidden)
        s_t = (
            h_cls.unsqueeze(1).expand(B, M, cfg.d_hidden)
            + roles.unsqueeze(0).expand(B, M, cfg.d_hidden)
        )

        seq_len = H.size(1) if H is not None else 1
        e_init = torch.zeros_like(s_t)
        r_init = torch.zeros_like(s_t)

        state_t = AgentState(
            s=s_t,
            b=torch.zeros(B, M, cfg.num_classes, device=pooled.device),
            u=torch.ones(B, M, 1, device=pooled.device),
            p=torch.zeros(B, M, seq_len, device=pooled.device),
            e=e_init,
            alpha=torch.ones(B, M, cfg.num_classes, device=pooled.device),
        )
        state_t = self.updater(state_t, r_init, e_init)

        # rho: используем EMA или переданное значение
        if rho is None:
            rho = self.rho_ema.expand(B, M, 1)
        else:
            rho = rho.to(pooled.device)

        # ── Deliberation Loop ─────────────────────────────────────────────────
        b_all, u_all, h_all, m_all, p_ptr_all = [], [], [], [], []
        b_all.append(state_t.b)
        u_all.append(state_t.u)

        rounds_used = 0

        for t in range(cfg.max_rounds):
            rounds_used = t + 1

            # Evidence Pointers
            if cfg.abl_no_pointers or H is None:
                p = torch.ones(B, M, seq_len, device=pooled.device) / seq_len
                e_t = state_t.s
            else:
                p, e_t = self.evidence_extractor(state_t.s, H, mask)

            state_t.p = p
            state_t.e = e_t
            p_ptr_all.append(p)

            # Message Formulation
            if cfg.abl_no_messages:
                m_t = torch.zeros(
                    B, M, self.communication.W_k.in_features, device=pooled.device
                )
                limit = min(cfg.d_hidden, self.communication.W_k.in_features)
                m_t[..., :limit] = state_t.s[..., :limit]
            else:
                m_t = self.message_formulator(state_t.s, state_t.e, state_t.b, state_t.u)
            m_all.append(m_t)

            # Communication
            if cfg.abl_no_communication:
                r_t = torch.zeros_like(state_t.s)
            else:
                r_t = self.communication(state_t.s, m_t, state_t.b, state_t.e)
                gate = self.comm_gate(state_t.s)   # (B, M, 1)
                r_t = gate * r_t
                if self.training and cfg.comm_dropout > 0:
                    r_t = F.dropout(r_t, p=cfg.comm_dropout, training=True)

            # Halting signal
            h_t = self.halter(state_t.s)
            h_all.append(h_t)

            # [v2] TransformerAgentUpdater: cross-attn + gated evidence + FFN
            state_t = self.updater(state_t, r_t, e_t)
            b_all.append(state_t.b)
            u_all.append(state_t.u)

            # ── Confidence-based Early Stopping ──────────────────────────────
            if not self.training and cfg.early_stop_threshold is not None:
                with torch.no_grad():
                    tmp_p, _, _ = self.consensus(
                        state_t.b, state_t.u, rho, epoch_fraction=epoch_fraction
                    )
                    if tmp_p.max(dim=-1).values.mean().item() >= cfg.early_stop_threshold:
                        break

        # ── Финальный консенсус ───────────────────────────────────────────────
        if cfg.abl_no_poe:
            w_final = torch.ones(B, M, device=pooled.device) / M
            p_final = state_t.b.mean(dim=1)
            L_dom = torch.tensor(0.0, device=pooled.device)
        else:
            p_final, w_final, L_dom = self.consensus(
                state_t.b, state_t.u, rho, epoch_fraction=epoch_fraction
            )

        L_role = self.role_embeddings.get_role_loss()

        # ── [v2] Обновление EMA надёжности ───────────────────────────────────
        if self.training:
            self._update_rho(b_all, p_final.detach())

        return {
            "p_final":      p_final,
            "b_all":        b_all,
            "u_all":        u_all,
            "m_all":        m_all,
            "p_ptr_all":    p_ptr_all,
            "h_all":        h_all,
            "L_dom":        L_dom,
            "L_role":       L_role,
            "state_final":  state_t,
            "w_final":      w_final,
            "rounds_used":  rounds_used,
            # EMA snapshot для мониторинга специализации агентов
            "rho_snapshot": self.rho_ema.squeeze().detach().cpu(),
        }
