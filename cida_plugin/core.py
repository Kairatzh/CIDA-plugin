"""
CIDAPlugin — Universal Evidence-Grounded Multi-Agent Deliberation Layer.
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
    PerspectiveProjector,
    AdaptiveDSchedule,
)
from .consensus import ConsensusAggregator, HaltingPredictor, ReliabilityTracker


class CIDAPlugin(nn.Module):
    """
    Универсальный CIDA-Plugin.

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
            cfg.d_hidden, cfg.num_classes, num_heads=cfg.num_attn_heads,
            multi_label=cfg.multi_label
        )

        # [v3+] Передаём normalize_components из конфига
        self.consensus = ConsensusAggregator(
            normalize_components=cfg.normalize_components
        )
        self.halter = HaltingPredictor(cfg.d_hidden, cfg.num_agents)

        # Gated communication warm-up
        self.comm_gate = nn.Sequential(
            nn.Linear(cfg.d_hidden, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.comm_gate[0].bias, -2.0)

        # ── [v3] Perspective Projector ────────────────────────────────────────
        if cfg.use_perspective_projector:
            self.perspective_projector = PerspectiveProjector(
                cfg.num_agents, cfg.d_input, cfg.d_hidden
            )

        # ── [v3+] Adaptive D-Schedule (PI-controller) ─────────────────────────
        if cfg.use_adaptive_d_schedule:
            self.adaptive_d_schedule = AdaptiveDSchedule(cfg.d_hidden)

        # ── [v3] Non-circular Reliability Tracker ─────────────────────────────
        self.reliability_tracker = ReliabilityTracker(cfg.num_agents, cfg.rho_decay)

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

    # NOTE: _update_rho (legacy v2 EMA) удалён.
    # Причина: circular dependency — надёжность считалась от согласия с
    # консенсусом, который сам зависит от надёжности. Заменён на
    # ReliabilityTracker (v3), который оценивает надёжность по accuracy.

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        pooled: torch.Tensor,
        seq_output: torch.Tensor = None,
        mask: torch.Tensor = None,
        rho: torch.Tensor = None,
        epoch_fraction: float = 0.0,
        y_true: torch.Tensor = None,
    ) -> dict:
        """
        Параметры
        ----------
        pooled      : (B, d_input)              pooled representation от энкодера
        seq_output  : (B, seq_len, d_input)     опционально — для evidence pointers
        mask        : (B, seq_len)              маска паддинга
        rho         : (B, M, 1) or None        переопределить EMA rho (для тестов)
        epoch_fraction : float                 прогресс обучения [0, 1]
        y_true      : (B,) or (B, K)           метки для обновления reliability

        Возвращает dict:
            p_final     : (B, K)      — финальное калиброванное распределение
            b_all       : list        — история верований агентов
            u_all       : list        — история неопределённостей
            m_all       : list        — история сообщений
            p_ptr_all   : list        — история evidence pointers
            h_all       : list        — история сигналов halting
            L_dom       : scalar      — штраф за доминирование
            L_role      : scalar      — штраф за несовместимость ролей
            L_orth      : scalar      — [v4] штраф за коллинеарность проекций
            state_final : AgentState  — финальное состояние
            w_final     : (B, M)      — веса агентов в консенсусе
            rounds_used : int         — фактическое число раундов
            ponder_cost : (B,)        — [v4] ожидаемое число раундов (ACT)
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
        
        if cfg.use_perspective_projector:
            # Каждому агенту своя проекция
            s_t = self.perspective_projector(pooled) + roles.unsqueeze(0)
        else:
            # Все агенты получают один и тот же pooled_output + свои роли
            s_t = (
                h_cls.unsqueeze(1).expand(B, M, cfg.d_hidden)
                + roles.unsqueeze(0).expand(B, M, cfg.d_hidden)
            )

        seq_len = H.size(1) if H is not None else 1
        e_init = torch.zeros_like(s_t)
        r_init = torch.zeros_like(s_t)

        if cfg.multi_label:
            b_init = torch.full((B, M, cfg.num_classes), 0.5, device=pooled.device)
        else:
            b_init = torch.zeros(B, M, cfg.num_classes, device=pooled.device)

        state_t = AgentState(
            s=s_t,
            b=b_init,
            u=torch.ones(B, M, 1, device=pooled.device),
            p=torch.zeros(B, M, seq_len, device=pooled.device),
            e=e_init,
            alpha=torch.ones(B, M, cfg.num_classes, device=pooled.device),
        )
        state_t = self.updater(state_t, r_init, e_init)

        # rho: используем v3 ReliabilityTracker или переданное значение
        if rho is None:
            rho = self.reliability_tracker.rho.expand(B, M, 1)
        else:
            rho = rho.to(pooled.device)

        # ── Deliberation Loop ─────────────────────────────────────────────────
        b_all, u_all, h_all, m_all, p_ptr_all, d_schedule = [], [], [], [], [], []

        rounds_used = 0

        # [v3+] Интегральный аккумулятор для PI-контроллера d-schedule
        pi_integral_acc = torch.zeros(B, 1, device=pooled.device)

        # [v4] ACT: аккумуляторы для взвешенного консенсуса
        if cfg.use_act_halting:
            act_remainder = torch.ones(B, 1, device=pooled.device)
            act_p_accum = torch.zeros(B, cfg.num_classes, device=pooled.device)
            act_w_accum = torch.zeros(B, M, device=pooled.device)
            act_L_dom_accum = torch.tensor(0.0, device=pooled.device)
            ponder_cost = torch.zeros(B, device=pooled.device)

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
            h_t = self.halter(state_t.s)   # (B, 1)
            h_all.append(h_t)

            # [v2] TransformerAgentUpdater: cross-attn + gated evidence + FFN
            state_t = self.updater(state_t, r_t, e_t)
            b_all.append(state_t.b)
            u_all.append(state_t.u)

            # [v4] ACT: накапливаем взвешенный консенсус
            if cfg.use_act_halting:
                # Промежуточный консенсус этого раунда
                p_t, w_t, L_dom_t = self.consensus(
                    state_t.b, state_t.u, rho,
                    epoch_fraction=epoch_fraction,
                    multi_label=cfg.multi_label,
                )

                if t < cfg.max_rounds - 1:
                    # Вес раунда = min(h_t, remainder) — не можем потратить больше остатка
                    w_round = torch.min(h_t, act_remainder)     # (B, 1)
                    act_remainder = act_remainder - w_round
                else:
                    # Последний раунд забирает весь остаток
                    w_round = act_remainder                      # (B, 1)

                act_p_accum = act_p_accum + w_round * p_t       # (B, K)
                act_w_accum = act_w_accum + w_round * w_t       # (B, M)
                act_L_dom_accum = act_L_dom_accum + w_round.mean() * L_dom_t
                ponder_cost = ponder_cost + w_round.squeeze(-1) * (t + 1)  # ожидаемый раунд

            # [v3+] Adaptive D-Schedule с PI-контроллером
            if cfg.use_adaptive_d_schedule:
                b_i = state_t.b.unsqueeze(2)
                b_j = state_t.b.unsqueeze(1)
                disag = torch.norm(b_i - b_j, p=1, dim=-1).mean(dim=(1, 2))  # (B,)
                disag = disag.unsqueeze(-1)  # (B, 1)

                d_t, error = self.adaptive_d_schedule(
                    h_cls, disag, integral_acc=pi_integral_acc
                )
                pi_integral_acc = pi_integral_acc + error
                d_schedule.append(d_t)

            # ── Confidence-based Early Stopping (inference only, non-ACT) ─────
            if not self.training and not cfg.use_act_halting:
                if cfg.early_stop_threshold is not None:
                    with torch.no_grad():
                        tmp_p, _, _ = self.consensus(
                            state_t.b, state_t.u, rho, epoch_fraction=epoch_fraction
                        )
                        if tmp_p.max(dim=-1).values.mean().item() >= cfg.early_stop_threshold:
                            break

        # ── Финальный консенсус ───────────────────────────────────────────────
        if cfg.use_act_halting:
            # [v4] ACT: p_final = взвешенная сумма консенсусов всех раундов
            p_final = act_p_accum
            w_final = act_w_accum
            L_dom = act_L_dom_accum
        elif cfg.abl_no_poe:
            w_final = torch.ones(B, M, device=pooled.device) / M
            p_final = state_t.b.mean(dim=1)
            L_dom = torch.tensor(0.0, device=pooled.device)
        else:
            p_final, w_final, L_dom = self.consensus(
                state_t.b, state_t.u, rho, epoch_fraction=epoch_fraction,
                multi_label=cfg.multi_label
            )

        L_role = self.role_embeddings.get_role_loss()
        
        # [v3] Специализированный ролевой лосс
        L_role_spec = torch.tensor(0.0, device=pooled.device)
        if cfg.use_explicit_roles and y_true is not None:
            L_role_spec = self.role_embeddings.get_role_specialization_loss(b_all, y_true)
            L_role_spec = L_role_spec * cfg.lambda_role_spec

        # [v4] Orthogonality loss
        L_orth = torch.tensor(0.0, device=pooled.device)
        if cfg.use_perspective_projector:
            L_orth = self.perspective_projector.orthogonality_loss() * cfg.lambda_orth

        # ── [v3] Обновление надёжности (нециклическое) ────────────────────────
        if self.training and y_true is not None:
            self.reliability_tracker.update(state_t.b, state_t.u, y_true)

        return {
            "p_final":      p_final,
            "b_all":        b_all,
            "u_all":        u_all,
            "m_all":        m_all,
            "p_ptr_all":    p_ptr_all,
            "h_all":        h_all,
            "L_dom":        L_dom,
            "L_role":       L_role,
            "L_role_spec":  L_role_spec,
            "L_orth":       L_orth,
            "state_final":  state_t,
            "w_final":      w_final,
            "rounds_used":  rounds_used,
            "d_schedule":   torch.cat(d_schedule, dim=1) if d_schedule else None,
            "ponder_cost":  ponder_cost if cfg.use_act_halting else None,
            "rho_snapshot": self.reliability_tracker.rho.squeeze().detach().cpu(),
        }

