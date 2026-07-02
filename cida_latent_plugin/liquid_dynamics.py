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

try:
    from torchdiffeq import odeint_adjoint as odeint_adj
    from torchdiffeq import odeint
    _HAS_ADJOINT = True
except ImportError:
    from torchdiffeq import odeint
    odeint_adj = odeint
    _HAS_ADJOINT = False

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
        T: float = 4.0,
    ):
        super().__init__()
        self.extractor = extractor
        self.message_formulator = message_formulator
        self.communication = communication
        self.updater = updater
        self.tau_net = LiquidTimeConstant(d_hidden)
        self.trajectory_save_every = trajectory_save_every
        self.T = T
        # Временные точки для сохранения (корректно для adaptive solvers)
        self._save_timepoints = {0.25, 0.50, 0.75, 1.0}

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
        self.nfe = 0
        # FIX v4: del [:] вместо clear() для явного освобождения памяти
        del self.b_trajectory[:]
        del self.u_trajectory[:]
        del self.p_ptr_trajectory[:]
        del self.m_trajectory[:]

    def clear_context(self):
        self.memory = None
        self.mask = None

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
        # FIX v4: 1) detach() — без него ODE граф держится в памяти (сотни МБ)
        #         2) temporal checkpoints вместо NFE-based (корректно для adaptive solvers)
        if torch.is_grad_enabled() or not self.training:
            t_normalized = t.item() / self.T if self.T > 0 else 0
            should_save = any(abs(t_normalized - sp) < 0.02 for sp in self._save_timepoints)
            if should_save or self.nfe % self.trajectory_save_every == 0:
                self.b_trajectory.append(next_state.b.detach())
                self.u_trajectory.append(next_state.u.detach())
                self.p_ptr_trajectory.append(p_ptr_t.detach())
                self.m_trajectory.append(s.detach())

        # TRUE Liquid ODE v4: τ(s,t) вычисляется на КАЖДОМ шаге из текущего s
        # (раньше τ фиксировался из s_init в set_context — это не настоящий Liquid NN)
        tau = self.tau_net(s)  # (B, M, 1) — state-dependent time constant
        f_s = next_state.s - s
        ds_dt = -s / tau + f_s

        return ds_dt


class LiquidDeliberationSolver(nn.Module):
    """
    Запускает ODE решатель для deliberation.
    
    При use_adjoint=True использует adjoint method для экономии VRAM:
    градиенты считаются без хранения промежуточных состояний.
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
        use_adjoint: bool = False,
    ):
        super().__init__()
        self.dynamics = LiquidDeliberationDynamics(
            extractor=extractor,
            message_formulator=message_formulator,
            communication=communication,
            updater=updater,
            d_hidden=d_hidden,
            trajectory_save_every=trajectory_save_every,
            T=T,
        )
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.T = T
        self.use_adjoint = use_adjoint and _HAS_ADJOINT
        self._integrate = odeint_adj if self.use_adjoint else odeint

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

        states = self._integrate(
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

