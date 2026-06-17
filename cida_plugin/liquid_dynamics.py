"""
Liquid Neural ODE Dynamics для CIDA-Plugin.

Заменяет дискретный цикл deliberation loop на непрерывное
дифференциальное уравнение:
    ds/dt = -s/τ(x) + F(s, r, e)

Где:
    s - скрытое состояние агентов
    r - контр-аргументы от других агентов
    e - извлечённые доказательства
    τ(x) - зависящая от входа константа времени (Liquid Time Constant)
    F - функция перехода (основанная на AgentUpdater)

Использование `torchdiffeq` позволяет использовать адаптивные решатели
(например, dopri5), которые автоматически регулируют количество
вычислений (NFE) в зависимости от сложности входа.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

from .agent import AgentState
from .deliberation import (
    AgentEvidenceExtractor,
    CounterargumentCommunication,
    AgentUpdater,
)


class LiquidTimeConstant(nn.Module):
    """
    Вычисляет τ(x) — константу времени, зависящую от начального состояния.
    """

    def __init__(self, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, s_init: torch.Tensor) -> torch.Tensor:
        """
        s_init: (B, M, d)
        Returns: (B, M, 1) — значения τ ∈ (0.1, 1.1)
        """
        return self.net(s_init) + 0.1


class LiquidDeliberationDynamics(nn.Module):
    """
    Обёртка над компонентами deliberation (extractor, communication, updater),
    которая вычисляет производную ds/dt для ODE решателя.

    Ключевое отличие от предыдущей версии: beliefs (b) и uncertainty (u)
    вычисляются ДИНАМИЧЕСКИ из текущего скрытого состояния s через
    updater.g(s), а не подставляются как нулевые заглушки.
    """

    def __init__(
        self,
        extractor: AgentEvidenceExtractor,
        message_formulator,
        communication: CounterargumentCommunication,
        updater: AgentUpdater,
        d_hidden: int,
        trajectory_save_every: int = 10,
    ):
        super().__init__()
        self.extractor = extractor
        self.message_formulator = message_formulator
        self.communication = communication
        self.updater = updater
        self.tau_net = LiquidTimeConstant(d_hidden)
        self.trajectory_save_every = trajectory_save_every

        # Кэшируемый контекст (чтобы не прокидывать через ODE)
        self.memory = None
        self.mask = None
        self.tau = None
        self.nfe = 0  # Number of Function Evaluations

        # Для сохранения промежуточных состояний (trajectory)
        self.b_trajectory = []
        self.u_trajectory = []
        self.p_ptr_trajectory = []
        self.m_trajectory = []

    def set_context(
        self,
        memory: torch.Tensor,
        mask: torch.Tensor,
        s_init: torch.Tensor,
    ):
        """
        Устанавливает контекст перед запуском ODE.
        memory: (B, L, d) — память документов
        mask: (B, L) — маска внимания
        s_init: (B, M, d) — начальное состояние агентов
        """
        self.memory = memory
        self.mask = mask
        self.tau = self.tau_net(s_init)  # (B, M, 1)
        self.nfe = 0
        self.b_trajectory.clear()
        self.u_trajectory.clear()
        self.p_ptr_trajectory.clear()
        self.m_trajectory.clear()

    def clear_context(self):
        self.memory = None
        self.mask = None
        self.tau = None

    def _compute_beliefs(self, s: torch.Tensor):
        """
        Динамически вычисляет текущие beliefs и uncertainty из состояния s.

        Вместо передачи dummy_b=0, dummy_u=1 (что лишало коммуникацию
        информации о текущих мнениях агентов), мы проецируем s через
        ту же голову, что использует AgentUpdater.

        Returns: (current_b, current_u)
        """
        g_out = self.updater.g(s)

        if self.updater.multi_label:
            current_b = torch.sigmoid(g_out)
            current_u = (4.0 * current_b * (1.0 - current_b)).mean(
                dim=-1, keepdim=True
            )
        else:
            alpha = F.softplus(g_out) + 1.0
            alpha_sum = alpha.sum(dim=-1, keepdim=True)
            current_b = alpha / alpha_sum
            current_u = float(g_out.size(-1)) / alpha_sum

        return current_b, current_u

    def forward(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет ds/dt.
        t: (1,) — текущее время ODE
        s: (B, M, d) — текущее состояние агентов
        """
        self.nfe += 1

        B, M = s.shape[0], s.shape[1]
        num_classes = self.updater.g[-1].out_features

        # Динамическое вычисление текущих beliefs и uncertainty
        current_b, current_u = self._compute_beliefs(s)

        # Evidence extraction
        p_ptr_t, e_t = self.extractor(s, self.memory, self.mask)

        # Communication с актуальными beliefs (не с нулями!)
        m_t = self.message_formulator(s, e_t, current_b, current_u)
        r_t = self.communication(s, m_t, current_b, e_t)

        # AgentUpdater step
        dummy_state = AgentState(
            s=s,
            b=current_b,
            u=current_u,
            p=torch.zeros(B, M, 1, device=s.device),
            e=e_t,
            alpha=torch.ones(B, M, num_classes, device=s.device),
        )
        next_state = self.updater(dummy_state, r_t, e_t)

        # Сохраняем промежуточные beliefs для лоссов и консенсуса
        # Сохраняем только каждое N-е состояние для экономии памяти с адаптивными решателями
        if torch.is_grad_enabled() or not self.training:
            if self.nfe % self.trajectory_save_every == 0:
                self.b_trajectory.append(next_state.b)
                self.u_trajectory.append(next_state.u)
                self.p_ptr_trajectory.append(p_ptr_t)
                self.m_trajectory.append(s)

        # Liquid ODE: ds/dt = -s/τ + f(s)
        f_s = next_state.s - s
        ds_dt = -s / self.tau + f_s

        return ds_dt


class LiquidDeliberationSolver(nn.Module):
    """
    Запускает ODE решатель для deliberation.
    """

    def __init__(
        self,
        extractor: AgentEvidenceExtractor,
        message_formulator,
        communication: CounterargumentCommunication,
        updater: AgentUpdater,
        d_hidden: int,
        solver: str = "euler",
        atol: float = 1e-3,
        rtol: float = 1e-3,
        T: float = 4.0,
        trajectory_save_every: int = 10,
    ):
        super().__init__()
        self.dynamics = LiquidDeliberationDynamics(
            extractor=extractor,
            message_formulator=message_formulator,
            communication=communication,
            updater=updater,
            d_hidden=d_hidden,
            trajectory_save_every=trajectory_save_every,
        )
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.T = T

    def forward(
        self,
        s_init: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple:
        """
        s_init: (B, M, d)
        memory: (B, L, d)
        mask: (B, L)

        Returns:
            s_final (B, M, d)
            trajectory (dict)
        """
        device = s_init.device
        self.dynamics.set_context(memory, mask, s_init)

        t = torch.tensor([0.0, self.T], device=device)

        states = odeint(
            self.dynamics,
            s_init,
            t,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
        )

        s_final = states[-1]
        nfe = self.dynamics.nfe

        trajectory = {
            "b_all": self.dynamics.b_trajectory,
            "u_all": self.dynamics.u_trajectory,
            "p_ptr_all": self.dynamics.p_ptr_trajectory,
            "m_all": self.dynamics.m_trajectory,
            "nfe": nfe,
        }

        self.dynamics.clear_context()

        return s_final, trajectory
