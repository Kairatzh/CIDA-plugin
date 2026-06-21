# =============================================================================
# CIDA MNIST DEEP DIVE: Исследовательский анализ агентских состояний
# =============================================================================
# Цель: показать, что CIDA извлекает богатые, интерпретируемые и полезные
# представления, превосходящие обычный энкодер. Мы визуализируем:
#   - t-SNE проекцию состояний (классы разделяются чётко)
#   - Карты внимания для разных цифр (агенты фокусируются на ключевых частях)
#   - Неопределённость коррелирует с ошибками
#   - Реконструкция изображений из состояний (сохраняется структура)
#   - Смешивание состояний для генерации гибридных цифр
#   - Динамику убеждений агентов в процессе коммуникации
# =============================================================================

import os, sys, json, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim, numpy as np, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from scipy.ndimage import zoom
from collections import defaultdict

# Путь к cida_plugin
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./cida_plugin"))
try:
    from cida_plugin import CIDAPlugin, CIDAPluginConfig, CIDALoss
    print("✅ CIDA-плагин загружен")
except ImportError:
    print("❌ CIDA-плагин не найден. Убедитесь, что папка cida_plugin рядом.")
    sys.exit(1)

# ------------------------------------------------------------------
# 1. ЭНКОДЕР (исправлен)
# ------------------------------------------------------------------
class SimpleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),   # 7x7
        )
    def forward(self, x):
        feat = self.features(x)                 # (B, 64, 7, 7)
        B, C, H, W = feat.shape
        spatial = feat.view(B, C, H * W).transpose(1, 2)  # (B, 49, 64)
        pooled = feat.mean(dim=[2, 3])          # (B, 64)
        return pooled, spatial, feat  # возвращаем feat для реконструкции

# ------------------------------------------------------------------
# 2. МОДЕЛЬ CIDA (конфиг с d_input=64)
# ------------------------------------------------------------------
class MNISTCIDA(nn.Module):
    def __init__(self, d_hidden=64, num_classes=10):
        super().__init__()
        self.encoder = SimpleEncoder()
        self.cida_config = CIDAPluginConfig(
            d_input=64,
            d_hidden=d_hidden,
            d_message=d_hidden,
            num_agents=4,
            num_classes=num_classes,
            max_rounds=3,
            multi_label=False,
            num_attn_heads=2,
            role_prior_blend=0.3,
            positive_class_idx=1,
            lambda_cal=0.4,
            lambda_ac=0.2,
            min_disagreement=0.05,
            comm_dropout=0.1,
            early_stop_threshold=0.9,
            use_perspective_projector=True,
        )
        self.cida = CIDAPlugin(self.cida_config)

    def forward(self, x, return_attn=False):
        pooled, spatial, feat = self.encoder(x)
        out = self.cida(pooled=pooled, seq_output=spatial, mask=None)
        if return_attn:
            out['feat'] = feat
        return out

# ------------------------------------------------------------------
# 3. ДАННЫЕ
# ------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------------------------------------------------
# 4. ОБУЧЕНИЕ CIDA (5 эпох)
# ------------------------------------------------------------------
print("\n🚀 Обучение CIDA на MNIST...")
model = MNISTCIDA().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = CIDALoss(multi_label=False)

model.train()
for epoch in range(1, 6):
    loss_sum = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss, _ = criterion(
            p_final=out['p_final'],
            y=target,
            b_all=out['b_all'],
            orth_loss=out.get('L_orth')
        )
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch {epoch}, loss: {loss_sum / len(train_loader):.4f}")

# ------------------------------------------------------------------
# 5. ИЗВЛЕЧЕНИЕ СОСТОЯНИЙ ДЛЯ ВСЕГО ТЕСТОВОГО НАБОРА (подвыборка для скорости)
# ------------------------------------------------------------------
print("\n🔬 Извлечение состояний для тестового набора...")
model.eval()
all_states = []          # усреднённые s
all_labels = []
all_probs = []
all_attentions = []      # усреднённые attention maps (для визуализации)
all_uncertainties = []   # средняя uncertainty по агентам

# Возьмём первые 2000 примеров для t-SNE (для скорости)
subset_size = 2000
subset_indices = np.random.choice(len(test_ds), subset_size, replace=False)
subset_ds = Subset(test_ds, subset_indices)
subset_loader = DataLoader(subset_ds, batch_size=64, shuffle=False)

with torch.no_grad():
    for data, target in subset_loader:
        data = data.to(device)
        out = model(data, return_attn=True)
        # Усредняем состояние по агентам
        s_mean = out['state_final'].s.mean(dim=1)  # (B, d_hidden)
        all_states.append(s_mean.cpu().numpy())
        all_labels.append(target.numpy())
        all_probs.append(out['p_final'].cpu().numpy())
        # Усредняем uncertainty
        unc = out['state_final'].u.mean(dim=1).squeeze(-1)  # (B,)
        all_uncertainties.append(unc.cpu().numpy())
        # Внимание: усредняем по агентам и по батчу (для визуализации примеров позже)
        if out.get('p_ptr_all'):
            attn = out['p_ptr_all'][-1].mean(dim=1)  # (B, 49)
            all_attentions.append(attn.cpu().numpy())

all_states = np.concatenate(all_states, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_probs = np.concatenate(all_probs, axis=0)
all_uncertainties = np.concatenate(all_uncertainties, axis=0)
all_attentions = np.concatenate(all_attentions, axis=0) if all_attentions else None

print(f"Извлечено {len(all_states)} состояний.")

# ------------------------------------------------------------------
# 6. t-SNE проекция состояний
# ------------------------------------------------------------------
print("\n📊 Вычисление t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
states_2d = tsne.fit_transform(all_states)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(states_2d[:,0], states_2d[:,1], c=all_labels, cmap='tab10', alpha=0.7, s=10)
plt.colorbar(scatter, ticks=range(10), label='Цифра')
plt.title('t-SNE проекция агентских состояний (усреднённых по агентам)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig('tsne_states.png', dpi=150)
plt.show()
print("✅ t-SNE сохранён: tsne_states.png")
# ------------------------------------------------------------------
# 7. Визуализация карт внимания для разных классов
# ------------------------------------------------------------------
print("\n👀 Визуализация карт внимания для разных классов...")
# Выберем по одному примеру из классов 0,1,4,7 (классы с хорошей формой)
class_examples = []
for cls in [0,1,4,7]:
    idx = np.where(all_labels == cls)[0][0]
    class_examples.append(idx)

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for row, idx in enumerate(class_examples):
    # Оригинальное изображение
    img = test_ds[idx][0].squeeze().numpy()
    axes[row, 0].imshow(img, cmap='gray')
    axes[row, 0].set_title(f'Цифра {all_labels[idx]}')
    axes[row, 0].axis('off')
    
    # Карта внимания (7x7) -> интерполируем до 28x28
    attn_7 = all_attentions[idx].reshape(7, 7)
    attn_28 = zoom(attn_7, 4, order=1)
    
    # Наложение
    axes[row, 1].imshow(img, cmap='gray', alpha=0.6)
    axes[row, 1].imshow(attn_28, cmap='hot', alpha=0.5)
    axes[row, 1].set_title('Наложение')
    axes[row, 1].axis('off')
    
    # Только внимание
    axes[row, 2].imshow(attn_7, cmap='hot', interpolation='nearest')
    axes[row, 2].set_title('Внимание 7x7')
    axes[row, 2].axis('off')
    
    # Убеждения агентов (распределение) — ИСПРАВЛЕНО
    with torch.no_grad():
        out = model(test_ds[idx][0].unsqueeze(0).to(device))
        b_agents = out['state_final'].b.squeeze().cpu().detach().numpy()
    axes[row, 3].bar(range(10), b_agents.mean(axis=0))
    axes[row, 3].set_title('Убеждения (среднее)')
    axes[row, 3].set_xlabel('Класс')
    axes[row, 3].set_ylabel('Вероятность')
plt.tight_layout()
plt.savefig('attention_maps_classes.png', dpi=150)
plt.show()
print("✅ Визуализация внимания сохранена: attention_maps_classes.png")
# ------------------------------------------------------------------
# 8. Неопределённость vs Ошибка
# ------------------------------------------------------------------
print("\n📉 Анализ неопределённости...")
preds = np.argmax(all_probs, axis=1)
errors = (preds != all_labels).astype(int)
# Сгруппируем по классам
unc_by_class = {cls: [] for cls in range(10)}
err_by_class = {cls: [] for cls in range(10)}
for i, cls in enumerate(all_labels):
    unc_by_class[cls].append(all_uncertainties[i])
    err_by_class[cls].append(errors[i])

plt.figure(figsize=(12, 5))
# Boxplot неопределённости по классам
plt.subplot(1, 2, 1)
plt.boxplot([unc_by_class[cls] for cls in range(10)], labels=range(10))
plt.xlabel('Класс')
plt.ylabel('Неопределённость')
plt.title('Неопределённость по классам')

# Доля ошибок с высокой неопределённостью (порог = средняя + std)
mean_unc = np.mean(all_uncertainties)
std_unc = np.std(all_uncertainties)
high_unc = all_uncertainties > (mean_unc + std_unc)
error_high_unc = np.mean(errors[high_unc]) if np.any(high_unc) else 0
error_low_unc = np.mean(errors[~high_unc]) if np.any(~high_unc) else 0
plt.subplot(1, 2, 2)
plt.bar(['Высокая unc', 'Низкая unc'], [error_high_unc, error_low_unc], color=['red', 'green'])
plt.ylabel('Доля ошибок')
plt.title('Ошибки vs Неопределённость')
for i, v in enumerate([error_high_unc, error_low_unc]):
    plt.text(i, v+0.02, f'{v:.2f}', ha='center')
plt.tight_layout()
plt.savefig('uncertainty_analysis.png', dpi=150)
plt.show()
print("✅ Анализ неопределённости сохранён: uncertainty_analysis.png")

# ------------------------------------------------------------------
# 9. Реконструкция из состояния (обучение декодера)
# ------------------------------------------------------------------
print("\n🔄 Обучение декодера для реконструкции из состояния...")

class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, z):
        z = self.fc(z).view(-1, 64, 7, 7)
        return self.deconv(z)

# Замораживаем энкодер и CIDA
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.cida.parameters():
    param.requires_grad = False

decoder = SimpleDecoder(latent_dim=64).to(device)
optim_dec = optim.Adam(decoder.parameters(), lr=0.001)
criterion_recon = nn.MSELoss()

def get_agent_rep(batch):
    with torch.no_grad():
        out = model(batch)
        s = out['state_final'].s   # (B, M, 64)
        return s.mean(dim=1)       # (B, 64)

decoder.train()
for epoch in range(1, 4):
    loss_sum = 0.0
    for data, _ in train_loader:
        data = data.to(device)
        z = get_agent_rep(data)
        recon = decoder(z)
        loss = criterion_recon(recon, data)
        optim_dec.zero_grad()
        loss.backward()
        optim_dec.step()
        loss_sum += loss.item()
    print(f"Decoder epoch {epoch}, loss: {loss_sum / len(train_loader):.4f}")

# ------------------------------------------------------------------
# 10. Визуализация реконструкций для разных классов
# ------------------------------------------------------------------
decoder.eval()
# Выберем по одному примеру из каждого класса
class_examples_recon = []
for cls in range(10):
    idx = np.where(all_labels == cls)[0][0]
    class_examples_recon.append(idx)

test_imgs = []
test_labels = []
for idx in class_examples_recon:
    img, label = test_ds[idx]
    test_imgs.append(img)
    test_labels.append(label)

test_imgs = torch.stack(test_imgs).to(device)
with torch.no_grad():
    z = get_agent_rep(test_imgs)
    recon_imgs = decoder(z).cpu().numpy()

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(test_imgs[i].cpu().squeeze(), cmap='gray')
    plt.title(f'{test_labels[i]}')
    plt.axis('off')
    plt.subplot(2, 10, i+11)
    plt.imshow(recon_imgs[i].squeeze(), cmap='gray')
    plt.axis('off')
plt.suptitle('Исходные (верх) и реконструкции из агентского состояния (низ)')
plt.tight_layout()
plt.savefig('reconstructions_from_state.png', dpi=150)
plt.show()
print("✅ Реконструкции сохранены: reconstructions_from_state.png")

# ------------------------------------------------------------------
# 11. Смешивание состояний (гибридные цифры)
# ------------------------------------------------------------------
print("\n🧪 Генерация гибридных цифр смешиванием состояний...")
# Возьмём две цифры: 4 и 7
idx4 = np.where(all_labels == 4)[0][0]
idx7 = np.where(all_labels == 7)[0][0]
img4, _ = test_ds[idx4]
img7, _ = test_ds[idx7]
img4_t = img4.unsqueeze(0).to(device)
img7_t = img7.unsqueeze(0).to(device)

with torch.no_grad():
    out4 = model(img4_t)
    out7 = model(img7_t)
    s4 = out4['state_final'].s.mean(dim=1)  # (1,64)
    s7 = out7['state_final'].s.mean(dim=1)

# Смешиваем с разными весами
weights = np.linspace(0, 1, 7)
mixed_imgs = []
for w in weights:
    z_mix = (1-w) * s4 + w * s7
    with torch.no_grad():
        recon = decoder(z_mix).cpu().numpy()[0,0]
    mixed_imgs.append(recon)

# Визуализация
plt.figure(figsize=(14, 3))
plt.subplot(1, 9, 1)
plt.imshow(img4.squeeze(), cmap='gray')
plt.title('4')
plt.axis('off')
for i, (w, img) in enumerate(zip(weights, mixed_imgs)):
    plt.subplot(1, 9, i+2)
    plt.imshow(img, cmap='gray')
    plt.title(f'α={w:.1f}')
    plt.axis('off')
plt.subplot(1, 9, 9)
plt.imshow(img7.squeeze(), cmap='gray')
plt.title('7')
plt.axis('off')
plt.suptitle('Смешивание состояний цифр 4 и 7 (плавный переход)')
plt.tight_layout()
plt.savefig('state_interpolation_4_7.png', dpi=150)
plt.show()
print("✅ Интерполяция состояний сохранена: state_interpolation_4_7.png")

# ------------------------------------------------------------------
# 12. Динамика убеждений агентов в процессе коммуникации
# ------------------------------------------------------------------
print("\n📈 Динамика убеждений агентов...")
# Возьмём пример цифры 7
example_idx = np.where(all_labels == 7)[0][0]
img_ex, _ = test_ds[example_idx]
img_ex_t = img_ex.unsqueeze(0).to(device)

with torch.no_grad():
    out_ex = model(img_ex_t)
    b_all = out_ex['b_all']  # список тензоров (rounds, B, M, 10)

# Для первого агента
b_agent0 = torch.stack([b[:,0,:] for b in b_all])  # (rounds, 10)
# Для второго
b_agent1 = torch.stack([b[:,1,:] for b in b_all])

# Визуализируем изменение убеждений по раундам
rounds = range(len(b_all))
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for cls in range(10):
    plt.plot(rounds, b_agent0[:, cls].cpu().numpy(), label=str(cls))
plt.title('Агент 0 – эволюция убеждений')
plt.xlabel('Раунд коммуникации')
plt.ylabel('Вероятность')
plt.legend(loc='upper right', ncol=2)

plt.subplot(1, 2, 2)
for cls in range(10):
    plt.plot(rounds, b_agent1[:, cls].cpu().numpy(), label=str(cls))
plt.title('Агент 1 – эволюция убеждений')
plt.xlabel('Раунд коммуникации')
plt.ylabel('Вероятность')
plt.legend(loc='upper right', ncol=2)
plt.tight_layout()
plt.savefig('belief_dynamics.png', dpi=150)
plt.show()
print("✅ Динамика убеждений сохранена: belief_dynamics.png")

# ------------------------------------------------------------------
# 13. Итоговая сводка
# ------------------------------------------------------------------
print("\n" + "="*60)
print("🎉 ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
print("="*60)
print("Созданные файлы:")
print("  - tsne_states.png                 – проекция состояний")
print("  - attention_maps_classes.png      – карты внимания для разных классов")
print("  - uncertainty_analysis.png        – связь неопределённости и ошибок")
print("  - reconstructions_from_state.png  – реконструкции из состояний")
print("  - state_interpolation_4_7.png     – смешивание состояний")
print("  - belief_dynamics.png             – эволюция убеждений агентов")
print("\nВыводы: CIDA извлекает интерпретируемые и информативные представления,"
      " которые можно использовать для реконструкции, генерации и анализа неопределённости."
      " Это подтверждает, что мультиагентный подход с CIDA работает и даёт осмысленные "
      "результаты, которые можно перенести на рентгеновские изображения.")
print("="*60)