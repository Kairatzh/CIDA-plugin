import torch
import torch.nn as nn
import torch.nn.functional as F


class OmegaLossSystem(nn.Module):
    """
    [v2] CIDA-Plugin Loss System с обучаемым расписанием разногласий.

    Новое в v2:
        - Learnable D-Schedule: параметры d_targets обучаются через backprop.
          Модель сама решает, сколько разногласий нужно на каждом раунде.
          Инициализация: затухающая последовательность [0.8 → 0.2].
        - d_schedule в forward() теперь опциональный. Если None — используется
          внутреннее обучаемое расписание.
    """

    def __init__(self,
                 lambda_task: float = 1.0,
                 lambda_cal: float = 0.5,
                 lambda_deb: float = 1.0,
                 lambda_prog: float = 0.5,
                 lambda_budget: float = 0.1,
                 lambda_role: float = 0.1,
                 lambda_dom: float = 0.1,
                 max_rounds: int = 4,
                 learnable_d_schedule: bool = True,
                 multi_label: bool = False):
        super().__init__()
        self.l_task = lambda_task
        self.l_cal = lambda_cal
        self.l_deb = lambda_deb
        self.l_prog = lambda_prog
        self.l_budget = lambda_budget
        self.l_role = lambda_role
        self.l_dom = lambda_dom
        self.multi_label = multi_label

        # ── Learnable D-Schedule ──────────────────────────────────────────────
        # Инициализируем: высокие разногласия в начале → низкие в конце.
        # Это отражает логику "сначала спорь, потом сходись".
        init_schedule = torch.linspace(0.8, 0.2, max_rounds)
        if learnable_d_schedule:
            # nn.Parameter → обучается через backprop вместе с остальными
            self.d_targets = nn.Parameter(init_schedule)
        else:
            # register_buffer → хранится в state_dict, но не обучается
            self.register_buffer("d_targets", init_schedule)

    def get_d_schedule(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Возвращает расписание разногласий для T раундов.
        Clamp гарантирует, что значения остаются в физически осмысленном диапазоне.
        """
        schedule = self.d_targets[:T]
        # Добиваем до T если d_targets короче (берём последнее значение)
        if len(schedule) < T:
            pad = schedule[-1].expand(T - len(schedule))
            schedule = torch.cat([schedule, pad])
        return torch.clamp(schedule, min=0.0, max=1.5).to(device)

    # ── Отдельные функции потерь ──────────────────────────────────────────────

    def task_loss(self, p_final: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Потери на основной задаче (CE или BCE)."""
        if self.multi_label:
            return F.binary_cross_entropy_with_logits(p_final, y)
        return F.cross_entropy(p_final, y, label_smoothing=0.05)

    def calibration_loss(self, p_final: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Brier score: MSE между вероятностями и метками."""
        if self.multi_label:
            # Для multi-label y уже (B, K)
            probs = torch.sigmoid(p_final)
            return torch.mean((probs - y) ** 2)
        
        y_one_hot = F.one_hot(y, num_classes=p_final.size(-1)).float()
        return torch.mean((p_final - y_one_hot) ** 2)

    def debate_loss(self, b_all: list, d_schedule: torch.Tensor) -> torch.Tensor:
        """
        Принуждает агентов к целевому уровню разногласий на каждом раунде.
        Ранние раунды: высокий d_t → агенты должны спорить.
        Поздние раунды: низкий d_t → агенты должны сходиться.

        [v2] d_schedule теперь Tensor, а не list — поддерживает градиенты
        от обучаемых d_targets.
        """
        T = len(b_all)
        if T == 0:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=b_all[0].device)
        for t in range(T):
            b_t = b_all[t]
            # [v3] Обработка d_schedule: может быть (T,) или (B, T)
            if d_schedule.dim() == 2:  # (B, T)
                d_t = d_schedule[:, t] if t < d_schedule.size(1) else d_schedule[:, -1]
            else:  # (T,)
                d_t = d_schedule[t] if t < len(d_schedule) else d_schedule[-1]

            b_i = b_t.unsqueeze(2)
            b_j = b_t.unsqueeze(1)
            disagreement = torch.norm(b_i - b_j, p=1, dim=-1).mean(dim=(1, 2))  # (B,)
            loss = loss + torch.mean((disagreement - d_t) ** 2)

        return loss

    def progress_loss(self, b_all: list, u_all: list, y: torch.Tensor,
                      margin: float = 0.05) -> torch.Tensor:
        """
        Штрафует «фиктивное рассуждение»: если агент не движется к правильному ответу.
        Учитывает uncertainty: высокая неопределённость → меньший штраф.
        """
        T = len(b_all)
        if T < 2:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=b_all[0].device)
        y_expanded = y.unsqueeze(1)

        for t in range(T - 1):
            b_t = b_all[t]
            b_next = b_all[t + 1]
            u_t = u_all[t].squeeze(-1)  # (B, M)

            B, M, K = b_t.shape
            
            if self.multi_label:
                # y: (B, K)
                ce_t = F.binary_cross_entropy(b_t, y.unsqueeze(1).expand(B, M, K), reduction='none').mean(dim=-1)
                ce_next = F.binary_cross_entropy(b_next, y.unsqueeze(1).expand(B, M, K), reduction='none').mean(dim=-1)
            else:
                ce_t = F.nll_loss(
                    torch.log(b_t.view(B * M, K) + 1e-9),
                    y_expanded.expand(B, M).reshape(B * M),
                    reduction='none'
                ).view(B, M)
                ce_next = F.nll_loss(
                    torch.log(b_next.view(B * M, K) + 1e-9),
                    y_expanded.expand(B, M).reshape(B * M),
                    reduction='none'
                ).view(B, M)

            dynamic_margin = margin * torch.clamp(1.0 - u_t, min=0.0)
            loss = loss + F.relu(ce_next - ce_t + dynamic_margin).mean()

        return loss

    def budget_loss(self, h_all: list) -> torch.Tensor:
        """ACT-style штраф за задержку остановки."""
        if not h_all:
            return torch.tensor(0.0)
        loss = torch.stack([h_t.mean() for h_t in h_all]).sum()
        return loss

    # ── Основной forward ─────────────────────────────────────────────────────

    def forward(self,
                p_final: torch.Tensor,
                y: torch.Tensor,
                b_all: list,
                u_all: list,
                h_all: list,
                l_dom_val: torch.Tensor,
                l_role_val: torch.Tensor,
                l_role_spec_val: torch.Tensor = torch.tensor(0.0),
                l_orth_val: torch.Tensor = torch.tensor(0.0),
                ponder_cost_val: torch.Tensor = None,
                d_schedule=None) -> tuple:
        """
        Параметры
        ----------
        p_final    : (B, K)         финальное распределение плагина
        y          : (B,)           истинные метки
        b_all      : list[(B,M,K)]  история верований агентов
        u_all      : list[(B,M,1)]  история неопределённостей
        h_all      : list[(B,1)]    история сигналов остановки
        l_dom_val  : scalar         штраф за доминирование
        l_role_val : scalar         штраф за несовместимость ролей
        l_role_spec_val: scalar     штраф за специализацию (взвешен)
        l_orth_val : scalar         штраф за коллинеарность (взвешен)
        ponder_cost_val : scalar/None ожидаемое число раундов (ACT)
        d_schedule : Tensor|list|None
        """
        device = p_final.device
        T = len(b_all)

        # Получаем расписание разногласий
        if d_schedule is None:
            sched = self.get_d_schedule(T, device)
        elif isinstance(d_schedule, list):
            sched = torch.tensor(d_schedule, device=device)
        else:
            sched = d_schedule

        l_t = self.task_loss(p_final, y)
        l_c = self.calibration_loss(p_final, y)
        l_d = self.debate_loss(b_all, sched)
        l_p = self.progress_loss(b_all, u_all, y)
        
        # Бюджет: либо по h_t (v1-v3), либо ponder_cost (v4 ACT)
        if ponder_cost_val is not None:
            l_b = ponder_cost_val.mean()
        else:
            l_b = self.budget_loss(h_all)

        total = (self.l_task * l_t
                 + self.l_cal * l_c
                 + self.l_deb * l_d
                 + self.l_prog * l_p
                 + self.l_budget * l_b
                 + self.l_dom * l_dom_val
                 + self.l_role * l_role_val
                 + l_role_spec_val
                 + l_orth_val)

        components = {
            "task":   l_t.item(),
            "cal":    l_c.item(),
            "deb":    l_d.item() if isinstance(l_d, torch.Tensor) else l_d,
            "prog":   l_p.item() if isinstance(l_p, torch.Tensor) else l_p,
            "budget": l_b.item() if isinstance(l_b, torch.Tensor) else l_b,
            "dom":    l_dom_val.item() if isinstance(l_dom_val, torch.Tensor) else l_dom_val,
            "role":   l_role_val.item() if isinstance(l_role_val, torch.Tensor) else l_role_val,
            "role_spec": l_role_spec_val.item() if isinstance(l_role_spec_val, torch.Tensor) else l_role_spec_val,
            "orth":   l_orth_val.item() if isinstance(l_orth_val, torch.Tensor) else l_orth_val,
            "d_schedule": sched.detach().cpu().tolist(),
        }
        return total, components
