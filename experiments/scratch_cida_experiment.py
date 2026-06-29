import torch
import torch.nn as nn
import torch.optim as optim
from cida_plugin import CIDAPluginConfig, CIDAPlugin

# ==========================================
# ЭКСПЕРИМЕНТ 2: CIDA ВСТРЕЧАЕТСЯ С НЕИЗВЕСТНОСТЬЮ (Dirichlet)
# ==========================================
# Обучаем ИИ-консилиум CIDA правилу: "Если X * Y > 25, Класс 1, иначе Класс 0".
# Включаем multi_label=False (Dirichlet), чтобы заработало полноценное разделение
# Epistemic (Незнание структуры) и Aleatoric (Шум данных / близость к границе).

torch.manual_seed(42)

cfg = CIDAPluginConfig(
    d_input=2,
    d_hidden=64,
    d_message=64,
    num_agents=4,
    num_classes=2,
    multi_label=False, # Включаем Dirichlet
    max_rounds=3,
    use_liquid_dynamics=False
)
plugin = CIDAPlugin(cfg)

# 1. Готовим данные (Только числа от 0 до 10)
X_train_raw = torch.rand(500, 2) * 10 
y_train = ((X_train_raw[:, 0] * X_train_raw[:, 1]) > 25).long()

# Нормализуем входы (Mean=0, Std=1) для стабильности обучения
mean = X_train_raw.mean(dim=0, keepdim=True)
std = X_train_raw.std(dim=0, keepdim=True)
X_train = (X_train_raw - mean) / (std + 1e-8)

optimizer = optim.Adam(plugin.parameters(), lr=0.002)
criterion = nn.NLLLoss()

print("Обучаем ИИ-консилиум CIDA (Dirichlet/Single-Label)...")
for epoch in range(600):
    optimizer.zero_grad()
    out = plugin(pooled=X_train) 
    
    # Классификационные потери + ортогонализация (anti-collapse)
    loss_cls = criterion(torch.log(out["p_final"] + 1e-10), y_train)
    loss = loss_cls + 0.1 * out["L_orth"]
    
    loss.backward()
    # Градиентный клиппинг против взрывов градиентов
    nn.utils.clip_grad_norm_(plugin.parameters(), 1.0)
    optimizer.step()
    
    if epoch % 100 == 0:
         # Вычисляем точность
         preds_classes = out["p_final"].argmax(dim=-1)
         acc = (preds_classes == y_train).float().mean().item()
         print(f"  Эпоха {epoch:3d} | Ошибка: {loss_cls.item():.4f} (L_orth: {out['L_orth'].item():.4f}) | Точность: {acc*100:.1f}%")

print("\nОбучение завершено. Проверяем на знакомых данных:")
plugin.eval()
with torch.no_grad():
    known_raw = torch.tensor([[8.0, 8.0], [2.0, 2.0]])
    known = (known_raw - mean) / std
    out_known = plugin(pooled=known)
    preds = out_known["p_final"]
    
    print(f"  X=8, Y=8 (Ожидаем Класс 1): Вероятность = {preds[0, 1].item():.4f} | Незнание (Epistemic U) = {out_known['u_epi'][0].mean().item():.4f} | Шум (Aleatoric U) = {out_known['u_alea'][0].mean().item():.4f}")
    print(f"  X=2, Y=2 (Ожидаем Класс 0): Вероятность = {preds[1, 1].item():.4f} | Незнание (Epistemic U) = {out_known['u_epi'][1].mean().item():.4f} | Шум (Aleatoric U) = {out_known['u_alea'][1].mean().item():.4f}")

print("\nЭКСПЕРИМЕНТ: Даем аномалии, которых ИИ никогда не видел!")
with torch.no_grad():
    # 100*100 (Вне диапазона, очевидный Класс 1, но OOD)
    # -50*-50 (Вне диапазона, отрицательные числа, OOD)
    # 5*5.1 (На границе принятия решений 25.5, должно быть много Aleatoric шума)
    ood_raw = torch.tensor([[100.0, 100.0], [-50.0, -50.0], [5.0, 5.1]])
    ood = (ood_raw - mean) / std
    out_ood = plugin(pooled=ood)
    preds_ood = out_ood["p_final"]
    
    print(f"  X=100, Y=100 (OOD): Вероятность Класса 1 = {preds_ood[0, 1].item():.4f} | Незнание (Epistemic U) = {out_ood['u_epi'][0].mean().item():.4f} | Шум (Aleatoric U) = {out_ood['u_alea'][0].mean().item():.4f}")
    print(f"  X=-50, Y=-50 (OOD): Вероятность Класса 1 = {preds_ood[1, 1].item():.4f} | Незнание (Epistemic U) = {out_ood['u_epi'][1].mean().item():.4f} | Шум (Aleatoric U) = {out_ood['u_alea'][1].mean().item():.4f}")
    print(f"  X=5, Y=5.1  (Граница): Вероятность Класса 1 = {preds_ood[2, 1].item():.4f} | Незнание (Epistemic U) = {out_ood['u_epi'][2].mean().item():.4f} | Шум (Aleatoric U) = {out_ood['u_alea'][2].mean().item():.4f}")

    print("\nРазногласие агентов (Disagreement) на аномалиях:")
    print(f"  X=100, Y=100: Спор = {out_ood['disagreement'][0].item():.4f}")
    print(f"  X=-50, Y=-50: Спор = {out_ood['disagreement'][1].item():.4f}")
    print(f"  X=5, Y=5.1  : Спор = {out_ood['disagreement'][2].item():.4f}")
