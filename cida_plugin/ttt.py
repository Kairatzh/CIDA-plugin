"""
Test-Time Training (TTT) для CIDA-Plugin.

Концепция (Apple 2024, MIT 2024):
    Обычно: обучение = оффлайн, инференс = заморожен.
    TTT: агент делает K шагов градиентного спуска на каждом входе,
    адаптируясь к конкретному примеру ПЕРЕД ответом.

    θ*_i = θ_i − α · ∇L_self(θ_i, x)   ← K шагов Adam
    b_i  = Agent_i(x; θ*_i)              ← belief после адаптации
    θ_i  ← θ_i                           ← восстановление весов

Почему это важно для CIDA:
    - Первая архитектура где агенты РЕАЛЬНО «думают» о каждом входе
    - Добавляем вычисление (K шагов), не параметры
    - Маленькая модель адаптируется как большая

Self-supervised loss:
    Masked reconstruction — маскируем часть компонент скрытого состояния
    агента и заставляем его восстанавливать замаскированное.
    Это заставляет агента «понять» структуру конкретного входа.
"""

import copy
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import AgentState


class MaskedReconstructionHead(nn.Module):
    """
    Вспомогательная голова для self-supervised loss.
    
    Принимает замаскированное скрытое состояние агента,
    предсказывает оригинальные значения замаскированных позиций.
    
    Архитектура: 2-layer MLP с bottleneck (d_hidden → d_hidden//2 → d_hidden).
    Bottleneck предотвращает тривиальное копирование и заставляет
    модель извлекать семантическую структуру из незамаскированных позиций.
    """

    def __init__(self, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, d_hidden),
        )

    def forward(self, masked_states: torch.Tensor) -> torch.Tensor:
        """
        masked_states: (B, M, d_hidden) — состояния с замаскированными компонентами.
        Returns:       (B, M, d_hidden) — предсказание оригинальных значений.
        """
        return self.net(masked_states)


class TestTimeTrainer(nn.Module):
    """
    Test-Time Training для агентов CIDA.
    
    При каждом forward-проходе:
    1. Создаёт маску (mask_ratio компонент обнуляются)
    2. Пропускает замаскированный вход через AgentUpdater
    3. Считает reconstruction loss на замаскированных позициях
    4. Делает K шагов Adam на updater + reconstruction_head
    5. После deliberation — восстанавливает оригинальные веса

    Параметры
    ---------
    d_hidden : int
        Размерность скрытых состояний агентов.
    ttt_steps : int
        Число шагов градиентного спуска (K). По умолчанию 3.
        Компромисс: больше шагов = лучше адаптация, но дороже.
    ttt_lr : float
        Learning rate для inner Adam. По умолчанию 1e-3.
        Должен быть достаточно большим для быстрой адаптации
        за K шагов, но не настолько чтобы сломать представления.
    mask_ratio : float
        Доля маскируемых компонент. По умолчанию 0.15 (как в BERT).
    """

    def __init__(
        self,
        d_hidden: int,
        ttt_steps: int = 3,
        ttt_lr: float = 1e-3,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.ttt_steps = ttt_steps
        self.ttt_lr = ttt_lr
        self.mask_ratio = mask_ratio
        self.d_hidden = d_hidden

        # Reconstruction head — обучается вместе с основной моделью
        self.reconstruction_head = MaskedReconstructionHead(d_hidden)

    def _create_mask(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Создаёт бинарную маску: 1 = сохранить, 0 = замаскировать.
        
        Маска одинаковая для всех агентов в батче (per-example, not per-agent)
        чтобы все агенты видели одинаковые «пробелы» и были вынуждены
        договариваться о восстановлении.
        
        shape: (B, M, d_hidden)
        Returns: (B, 1, d_hidden) — broadcast по агентам
        """
        B = shape[0]
        d = shape[2]
        num_masked = max(1, int(d * self.mask_ratio))

        mask = torch.ones(B, 1, d, device=device)
        for i in range(B):
            indices = torch.randperm(d, device=device)[:num_masked]
            mask[i, 0, indices] = 0.0

        return mask

    def compute_reconstruction_loss(
        self,
        updater: nn.Module,
        masked_states: torch.Tensor,
        original_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Self-supervised loss: MSE на замаскированных позициях.

        updater         : nn.Module — AgentUpdater для проброса замаскированных состояний
        masked_states   : (B, M, d_hidden) — состояния с маской (вход)
        original_states : (B, M, d_hidden) — оригинальные состояния (target)
        mask            : (B, 1, d_hidden) — маска (0 = замаскировано)

        Returns: scalar — средний MSE на замаскированных позициях.
        """
        B, M, d = masked_states.shape
        device = masked_states.device
        
        # Создаем фиктивные данные для AgentUpdater (ему нужны только s, e, r для forward)
        dummy_state = AgentState(
            s=masked_states,
            b=torch.zeros(B, M, 1, device=device),
            u=torch.ones(B, M, 1, device=device),
            p=torch.zeros(B, M, 1, device=device),
            e=torch.zeros(B, M, d, device=device),
            alpha=torch.zeros(B, M, 1, device=device)
        )
        dummy_r = torch.zeros(B, M, d, device=device)
        dummy_e = torch.zeros(B, M, d, device=device)
        
        # Пропускаем через updater (чтобы его веса получили градиенты!)
        updated_state = updater(dummy_state, dummy_r, dummy_e)
        
        predicted = self.reconstruction_head(updated_state.s)  # (B, M, d)
        inverted_mask = 1.0 - mask  # 1 = замаскированные позиции

        # MSE только на замаскированных
        diff = (predicted - original_states) ** 2
        masked_diff = diff * inverted_mask  # (B, M, d)

        # Средний loss: сумма по маскированным / число маскированных
        num_masked = inverted_mask.sum().clamp(min=1.0)
        loss = masked_diff.sum() / num_masked

        return loss

    @contextmanager
    def adapt_context(
        self,
        updater: nn.Module,
        agent_states: torch.Tensor,
    ):
        """
        Context manager для TTT-адаптации.

        Внутри контекста:
            - updater имеет адаптированные веса (θ*)
            - reconstruction_head тоже адаптирован
        При выходе:
            - Все веса восстанавливаются к θ

        Использование:
            with ttt.adapt_context(updater, s_t) as ttt_info:
                # updater уже адаптирован, запускаем deliberation
                state_t = updater(state_t, r_t, e_t)

        Возвращает dict с диагностикой:
            - 'losses': list[float] — losses по шагам
            - 'param_shift': float — ||θ* - θ||
            - 'steps_done': int — фактическое число шагов
        
        Parameters
        ----------
        updater : nn.Module
            AgentUpdater, чьи веса будут адаптированы.
        agent_states : torch.Tensor
            (B, M, d_hidden) — текущие состояния агентов (target для реконструкции).
        """
        original_states = agent_states.detach()
        device = agent_states.device

        # ── Сохраняем оригинальные параметры ────────────────────────────
        saved_updater_state = copy.deepcopy(updater.state_dict())
        saved_head_state = copy.deepcopy(self.reconstruction_head.state_dict())

        # ── Создаём маску ───────────────────────────────────────────────
        mask = self._create_mask(agent_states.shape, device)

        # ── Inner optimization (K шагов Adam) ───────────────────────────
        # Оптимизируем только updater + reconstruction_head
        inner_params = list(updater.parameters()) + list(
            self.reconstruction_head.parameters()
        )
        inner_optimizer = torch.optim.Adam(inner_params, lr=self.ttt_lr)

        losses = []
        for step in range(self.ttt_steps):
            inner_optimizer.zero_grad()

            # Маскируем состояния
            masked_states = original_states * mask  # (B, M, d)

            # Reconstruction loss
            loss = self.compute_reconstruction_loss(
                updater, masked_states, original_states, mask
            )

            loss.backward(retain_graph=(step < self.ttt_steps - 1))
            inner_optimizer.step()
            losses.append(loss.item())

        # ── Вычисляем сдвиг параметров ──────────────────────────────────
        param_shift = 0.0
        current_state = updater.state_dict()
        for key in saved_updater_state:
            if key in current_state:
                param_shift += (
                    current_state[key].float() - saved_updater_state[key].to(device).float()
                ).norm(2).item()

        ttt_info = {
            "losses": losses,
            "param_shift": param_shift,
            "steps_done": self.ttt_steps,
        }

        try:
            yield ttt_info
        finally:
            # ── Восстанавливаем оригинальные веса ───────────────────────
            updater.load_state_dict(saved_updater_state)
            self.reconstruction_head.load_state_dict(saved_head_state)

    def adapt_and_get_info(
        self,
        updater: nn.Module,
        agent_states: torch.Tensor,
    ) -> dict:
        """
        Неконтекстная версия: адаптирует updater и возвращает диагностику.
        Веса НЕ восстанавливаются — вызывающий код должен позаботиться.

        Полезно когда нужно fine-grained контроль или когда контекст
        менеджер неудобен (например внутри ODE-solver).

        Returns: dict с 'losses', 'param_shift', 'steps_done',
                 'saved_updater_state', 'saved_head_state'.
        """
        original_states = agent_states.detach()
        device = agent_states.device

        saved_updater_state = copy.deepcopy(updater.state_dict())
        saved_head_state = copy.deepcopy(self.reconstruction_head.state_dict())

        mask = self._create_mask(agent_states.shape, device)

        inner_params = list(updater.parameters()) + list(
            self.reconstruction_head.parameters()
        )
        inner_optimizer = torch.optim.Adam(inner_params, lr=self.ttt_lr)

        losses = []
        for step in range(self.ttt_steps):
            inner_optimizer.zero_grad()
            masked_states = original_states * mask
            loss = self.compute_reconstruction_loss(
                updater, masked_states, original_states, mask
            )
            loss.backward(retain_graph=(step < self.ttt_steps - 1))
            inner_optimizer.step()
            losses.append(loss.item())

        param_shift = 0.0
        current_state = updater.state_dict()
        for key in saved_updater_state:
            if key in current_state:
                param_shift += (
                    current_state[key].float() - saved_updater_state[key].to(device).float()
                ).norm(2).item()

        return {
            "losses": losses,
            "param_shift": param_shift,
            "steps_done": self.ttt_steps,
            "saved_updater_state": saved_updater_state,
            "saved_head_state": saved_head_state,
        }

    def restore_weights(
        self,
        updater: nn.Module,
        saved_updater_state: dict,
        saved_head_state: dict,
    ):
        """Восстановить оригинальные веса после TTT-адаптации."""
        updater.load_state_dict(saved_updater_state)
        self.reconstruction_head.load_state_dict(saved_head_state)
