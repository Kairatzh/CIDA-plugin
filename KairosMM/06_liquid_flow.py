"""
06_liquid_flow.py — Liquid Flow Deliberation (CIDA + Flow Matching).

РЕВОЛЮЦИОННАЯ АРХИТЕКТУРА (Meta/MIT 2026 + Liquid Neural ODE).
Вместо обычного ODE или SDE мы решаем объединённое дифференциальное уравнение:

    ds/dt = -s/τ(x) + F_agents(s,r,e) + λ·v_θ(s,t)

Где:
  -s/τ(x)       : Liquid Time Constant (забывание старых состояний)
  F_agents(...) : Взаимодействие агентов (Proposer, Skeptic и др.)
  v_θ(s,t)      : Flow Matching velocity field (направляет к консенсусу)

Flow Matching обучает v_θ(s,t) предсказывать векторное поле от шума к данным 
по прямой линии (ODE), что в 10 раз быстрее обычной диффузии.

Использование:
    python KairosMM/06_liquid_flow.py
"""

import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    print("[WARN] torchdiffeq не установлен, используем простую Euler интеграцию")
    def odeint(func, y0, t, method='euler', **kwargs):
        # Простой Euler solver fallback
        dt = t[1] - t[0]
        y = y0
        for _ in range(int(t[-1]/dt)):
            y = y + func(t[0], y) * dt
        return torch.stack([y0, y])


class FlowVelocityField(nn.Module):
    """
    v_θ(s, t) — векторное поле для Flow Matching.
    Предсказывает направление изменения латентного состояния s 
    в момент времени t.
    """
    def __init__(self, d_hidden: int):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden)
        )
        self.net = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden * 2),
            nn.SiLU(),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.LayerNorm(d_hidden)
        )

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        s: (B, M, d_hidden)
        t: (1,) или скаляр
        """
        B, M, D = s.shape
        # Time embedding
        t_tensor = t.view(1, 1).expand(B, 1) if t.dim() == 0 else t.view(1, -1).expand(B, 1)
        t_emb = self.time_embed(t_tensor).unsqueeze(1).expand(-1, M, -1) # (B, M, D)
        
        # Combine state and time
        x = torch.cat([s, t_emb], dim=-1)
        v = self.net(x)
        return v


class LiquidFlowDynamics(nn.Module):
    """
    Объединённая динамика: Liquid Time-Constant ODE + Flow Matching ODE.
    ds/dt = -s/τ + F_agents + λ*v_θ
    """
    def __init__(
        self, 
        d_hidden: int, 
        lambda_flow: float = 0.5,
        tau_base: float = 1.0
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.lambda_flow = lambda_flow
        self.tau = tau_base
        
        # Flow Matching Vector Field
        self.flow_v = FlowVelocityField(d_hidden)
        
        # Dummy F_agents (в реальной интеграции сюда подаётся AgentUpdater)
        self.agent_interaction = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh()
        )

    def forward(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        t: (1,) — время интеграции [0, 1]
        s: (B, M, d_hidden) — латентное состояние агентов
        """
        # 1. Liquid component: забывание
        liquid_decay = -s / self.tau
        
        # 2. Agent deliberation (в реальном CIDA — cross-attention)
        f_agents = self.agent_interaction(s)
        
        # 3. Flow Matching Velocity: направляет траекторию к данным
        flow_velocity = self.flow_v(s, t)
        
        # Полная динамика
        ds_dt = liquid_decay + f_agents + self.lambda_flow * flow_velocity
        return ds_dt


def train_flow_matching_step(
    model: LiquidFlowDynamics, 
    optimizer: torch.optim.Optimizer,
    s_target: torch.Tensor, 
):
    """
    Flow Matching Training (Simulation-free!).
    В отличие от обычной диффузии, мы обучаем векторное поле интерполировать 
    по ПРЯМОЙ линии между шумом s_0 и данными s_1.
    """
    B, M, D = s_target.shape
    device = s_target.device
    
    # 1. Сэмплим шум
    s_0 = torch.randn_like(s_target)
    
    # 2. Сэмплим случайное время t in [0, 1]
    t = torch.rand(B, 1, 1, device=device)
    
    # 3. Вычисляем промежуточную точку на прямой (t=0 -> s_0, t=1 -> s_1)
    s_t = (1 - t) * s_0 + t * s_target
    
    # 4. Истинное векторное поле для прямой линии (constant velocity)
    v_true = s_target - s_0
    
    # 5. Предсказание модели
    # t подаём как скаляр для простоты в этом демо
    t_scalar = t.mean()
    v_pred = model.flow_v(s_t, t_scalar)
    
    # 6. Loss = MSE между предсказанной и истинной скоростью
    loss = F.mse_loss(v_pred, v_true)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def demo_liquid_flow():
    print("="*60)
    print("🌊 Liquid Flow Deliberation Demo (Flow Matching + ODE)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, M, D = 4, 6, 768  # 4 батч, 6 агентов, 768 дим
    
    model = LiquidFlowDynamics(d_hidden=D).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    s_target = torch.randn(B, M, D, device=device)
    
    print("1. Обучение Flow Matching (Simulation-Free)...")
    for step in range(50):
        loss = train_flow_matching_step(model, optimizer, s_target)
        if step % 10 == 0:
            print(f"   Step {step:02d} | Flow Matching Loss: {loss:.4f}")
            
    print("\n2. Inference (ODE Integration)...")
    model.eval()
    with torch.no_grad():
        s_init = torch.randn(B, M, D, device=device)
        t_span = torch.tensor([0.0, 1.0], device=device)
        
        # Решаем ODE (t=0 -> t=1)
        print("   Запуск ODE solver (dopri5 / euler)...")
        trajectory = odeint(model, s_init, t_span, method='euler')
        s_final = trajectory[-1]
        
    print(f"\n   s_init shape : {s_init.shape}")
    print(f"   s_final shape: {s_final.shape}")
    shift = torch.norm(s_final - s_init).item()
    print(f"   L2 Shift (Deliberation traveled distance): {shift:.2f}")
    print("\n✅ Liquid Flow готов к интеграции в core.py!")

if __name__ == "__main__":
    demo_liquid_flow()
