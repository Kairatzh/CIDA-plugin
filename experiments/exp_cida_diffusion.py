"""
CIDA Kairos — DiffusionGemma + Liquid Neural ODE Integration.

Архитектура "Continuous-Discrete Integration":
    1. DiffusionGemma берёт шумный canvas (256 токенов) и итеративно
       очищает его через дискретную диффузию.
    2. На каждом N-м шаге диффузии скрытые состояния перехватываются
       через forward hooks и подаются в LiquidDeliberationSolver.
    3. Агенты CIDA "обсуждают" текущий набросок кода в непрерывном
       времени (Liquid ODE) и возвращают уточнённый embedding.
    4. Уточнённый embedding возвращается в DiffusionGemma для
       направления следующего шага диффузии.

Оптимизации для облака (Kaggle T4 / Colab A100):
    - Unsloth: 2-5x ускорение, -70% VRAM
    - QLoRA (4-bit NF4): модель занимает ~18GB вместо ~52GB
    - odeint_adjoint: градиенты ODE без хранения графа
    - gradient_checkpointing: экономия памяти при backprop

Запуск:
    python experiments/exp_cida_diffusion.py --dry_run   # тест на CPU
    python experiments/exp_cida_diffusion.py              # полный запуск GPU

Требования:
    pip install unsloth torchdiffeq bitsandbytes accelerate peft
"""

import argparse
import sys
import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Добавляем корень проекта в sys.path ────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cida_plugin.config import CIDAPluginConfig
from cida_plugin.core import CIDAPlugin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cida_kairos")


# ═════════════════════════════════════════════════════════════════════════════
# 1. КОНФИГУРАЦИЯ
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class KairosConfig:
    """Конфигурация CIDA Kairos эксперимента."""
    # ── Модель ──────────────────────────────────────────────────────────────
    model_id: str = "google/diffusiongemma-26B-A4B-it"
    max_seq_length: int = 512          # Контекст для обучения
    canvas_length: int = 256           # Размер canvas диффузии

    # ── CIDA Plugin ─────────────────────────────────────────────────────────
    cida_d_hidden: int = 256           # Уменьшенный d_hidden для экономии
    cida_num_agents: int = 4           # Prosecutor, Defender, Skeptic, Integrator
    cida_max_rounds: int = 3           # T для ODE
    cida_inject_every_n: int = 4       # Инъекция CIDA каждые N шагов диффузии

    # ── Обучение ────────────────────────────────────────────────────────────
    lora_r: int = 16                   # Ранг LoRA
    lora_alpha: int = 32               # Alpha LoRA
    learning_rate: float = 2e-4
    num_epochs: int = 1
    batch_size: int = 1
    grad_accum_steps: int = 4
    warmup_steps: int = 50
    max_steps: int = 500

    # ── Данные ──────────────────────────────────────────────────────────────
    dataset_id: str = "bigcode/starcoderdata"
    dataset_subset: str = "python"
    dataset_split: str = "train"
    max_samples: int = 2000            # Качество > количество


# ═════════════════════════════════════════════════════════════════════════════
# 2. CIDA DELIBERATION HOOK — Инъекция ODE в DiffusionGemma
# ═════════════════════════════════════════════════════════════════════════════

class CIDADiffusionBridge(nn.Module):
    """
    Мост между DiffusionGemma и CIDA Plugin.

    Перехватывает скрытые состояния модели через forward hooks на
    энкодерных слоях и пропускает их через LiquidDeliberationSolver.

    Архитектура:
        hidden_states (B, L, D_model)
            → proj_down (D_model → d_hidden)
            → CIDAPlugin (Liquid ODE deliberation)
            → proj_up (d_hidden → D_model)
            → residual addition к исходным hidden_states
    """

    def __init__(
        self,
        d_model: int,
        cida_config: CIDAPluginConfig,
        gate_init_bias: float = -2.0,
    ):
        super().__init__()
        d_hidden = cida_config.d_hidden

        # Проекции вниз/вверх для стыковки размерностей
        self.proj_down = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.LayerNorm(d_hidden),
        )
        self.proj_up = nn.Linear(d_hidden, d_model)

        # CIDA Plugin с Liquid ODE
        self.plugin = CIDAPlugin(cida_config)

        # Обучаемый gate: контролирует силу инъекции CIDA
        # Инициализируем с отрицательным bias → gate ≈ 0 в начале
        # → модель сначала учится без CIDA, потом постепенно включает
        self.gate = nn.Parameter(torch.tensor(gate_init_bias))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        hidden_states: (B, L, D_model) — из энкодера DiffusionGemma
        Returns: (B, L, D_model) — модифицированные hidden_states
        """
        B, L, D = hidden_states.shape

        # Проецируем вниз
        h_down = self.proj_down(hidden_states)  # (B, L, d_hidden)

        # Pooled representation (mean pooling)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (h_down * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
        else:
            pooled = h_down.mean(dim=1)  # (B, d_hidden)

        # Запускаем CIDA Plugin (Liquid ODE deliberation)
        cida_out = self.plugin(
            pooled=pooled,
            seq_output=h_down,
            mask=attention_mask,
        )

        # Результат deliberation → проекция обратно
        # Используем state_final.s (финальные состояния агентов)
        # и расширяем до размера последовательности через attention
        agent_states = cida_out["state_final"].s  # (B, M, d_hidden)

        # Cross-attention: каждый токен "спрашивает" агентов
        # Q: tokens, K/V: agents
        scores = torch.bmm(
            h_down, agent_states.transpose(1, 2)
        ) / (h_down.size(-1) ** 0.5)  # (B, L, M)
        attn_weights = F.softmax(scores, dim=-1)
        cida_signal = torch.bmm(attn_weights, agent_states)  # (B, L, d_hidden)

        # Проецируем обратно + gated residual
        delta = self.proj_up(cida_signal)  # (B, L, D_model)
        gate_value = torch.sigmoid(self.gate)

        return hidden_states + gate_value * delta


# ═════════════════════════════════════════════════════════════════════════════
# 3. HOOK MANAGER — Управление forward hooks на энкодере
# ═════════════════════════════════════════════════════════════════════════════

class DiffusionHookManager:
    """
    Менеджер forward hooks для инъекции CIDA в DiffusionGemma.

    Подключает CIDADiffusionBridge к указанным слоям энкодера.
    Hook вызывается на каждом forward pass, что при диффузии
    означает: на каждом шаге деноизинга.
    """

    def __init__(
        self,
        model: nn.Module,
        bridge: CIDADiffusionBridge,
        target_layer_idx: int = -1,
        inject_every_n: int = 4,
    ):
        self.bridge = bridge
        self.inject_every_n = inject_every_n
        self._step_counter = 0
        self._hooks: List = []
        self._attention_mask = None

        # Находим энкодерные слои
        layers = self._find_encoder_layers(model)
        if not layers:
            log.warning("Не найдены encoder layers! Hook не установлен.")
            return

        target_layer = layers[target_layer_idx]
        hook = target_layer.register_forward_hook(self._hook_fn)
        self._hooks.append(hook)
        log.info(
            f"CIDA hook установлен на слой {target_layer_idx} "
            f"(inject_every_n={inject_every_n})"
        )

    def set_attention_mask(self, mask: torch.Tensor):
        """Устанавливает маску для текущего батча."""
        self._attention_mask = mask

    def reset_step_counter(self):
        """Сбрасывает счетчик шагов диффузии."""
        self._step_counter = 0

    def _hook_fn(self, module, input, output):
        """Forward hook: перехватывает и модифицирует hidden_states."""
        self._step_counter += 1

        # Инъекция только каждые N шагов (экономия вычислений)
        if self._step_counter % self.inject_every_n != 0:
            return output

        # Определяем hidden_states из output
        if isinstance(output, tuple):
            hidden_states = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states = output
        else:
            return output

        # Применяем CIDA Bridge
        modified = self.bridge(hidden_states, self._attention_mask)

        # Возвращаем в том же формате
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    def _find_encoder_layers(self, model: nn.Module) -> list:
        """Эвристический поиск encoder layers в модели."""
        # DiffusionGemma / Gemma architecture
        for attr in ["model.layers", "layers", "encoder.layers",
                      "transformer.h", "model.model.layers"]:
            parts = attr.split(".")
            obj = model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                if hasattr(obj, "__len__") and len(obj) > 0:
                    return list(obj)
            except AttributeError:
                continue
        return []

    def remove_hooks(self):
        """Удаляет все hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ═════════════════════════════════════════════════════════════════════════════
# 4. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def build_cida_config(cfg: KairosConfig, d_model: int) -> CIDAPluginConfig:
    """Создаёт конфигурацию CIDA Plugin, оптимизированную для Kairos."""
    return CIDAPluginConfig(
        d_input=cfg.cida_d_hidden,
        d_hidden=cfg.cida_d_hidden,
        d_message=cfg.cida_d_hidden,
        num_agents=cfg.cida_num_agents,
        num_classes=2,  # binary: "хороший код" vs "плохой код"
        max_rounds=cfg.cida_max_rounds,
        use_liquid_dynamics=True,
        liquid_solver="euler",
        liquid_atol=1e-2,
        liquid_rtol=1e-2,
        trajectory_save_every=1,
        use_perspective_projector=True,
        use_ttt=False,  # Экономим ресурсы
        comm_dropout=0.1,
        early_stop_threshold=None,
    )


def load_model_and_tokenizer(cfg: KairosConfig, dry_run: bool = False):
    """Загружает DiffusionGemma через Unsloth с 4-bit квантованием."""

    if dry_run:
        log.info("=== DRY RUN: создаём фейковую модель ===")
        return _create_dummy_model(cfg)

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        log.error(
            "Unsloth не установлен! Установите: pip install unsloth\n"
            "Или используйте --dry_run для тестирования без GPU."
        )
        sys.exit(1)

    log.info(f"Загружаем {cfg.model_id} через Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_id,
        max_seq_length=cfg.max_seq_length,
        dtype=None,  # Авто-определение
        load_in_4bit=True,
        trust_remote_code=True, # Обязательно для новых архитектур как DiffusionGemma
    )

    log.info(f"Применяем LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth-оптимизация
    )

    return model, tokenizer


def load_dataset(cfg: KairosConfig, tokenizer, dry_run: bool = False):
    """Загружает и токенизирует датасет кода."""

    if dry_run:
        log.info("=== DRY RUN: создаём фейковый датасет ===")
        return _create_dummy_dataset(cfg)

    from datasets import load_dataset as hf_load

    log.info(f"Загружаем датасет: {cfg.dataset_id}/{cfg.dataset_subset}...")
    ds = hf_load(
        cfg.dataset_id,
        data_dir=cfg.dataset_subset,
        split=f"{cfg.dataset_split}[:{cfg.max_samples}]",
        streaming=False,
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["content"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding="max_length",
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")
    return ds


def train(cfg: KairosConfig, dry_run: bool = False):
    """Основной цикл обучения CIDA Kairos."""

    log.info("=" * 60)
    log.info("  CIDA KAIROS — DiffusionGemma + Liquid Neural ODE")
    log.info("=" * 60)

    # ── 1. Загрузка модели ──────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg, dry_run)

    # Определяем d_model
    if hasattr(model, "config"):
        d_model = getattr(model.config, "hidden_size", 2048)
    else:
        d_model = 2048
    log.info(f"D_model = {d_model}")

    # ── 2. Создаём CIDA Bridge ──────────────────────────────────────────────
    cida_config = build_cida_config(cfg, d_model)
    bridge = CIDADiffusionBridge(
        d_model=d_model,
        cida_config=cida_config,
        gate_init_bias=-2.0,
    )

    device = next(model.parameters()).device
    bridge = bridge.to(device)
    log.info(
        f"CIDA Bridge: {sum(p.numel() for p in bridge.parameters()):,} params "
        f"({sum(p.numel() for p in bridge.parameters() if p.requires_grad):,} trainable)"
    )

    # ── 3. Устанавливаем hooks ──────────────────────────────────────────────
    hook_mgr = DiffusionHookManager(
        model=model,
        bridge=bridge,
        target_layer_idx=-2,  # Предпоследний слой
        inject_every_n=cfg.cida_inject_every_n,
    )

    # ── 4. Загрузка данных ──────────────────────────────────────────────────
    dataset = load_dataset(cfg, tokenizer, dry_run)

    # ── 5. Настройка оптимизатора ───────────────────────────────────────────
    # Обучаем только: LoRA адаптеры + CIDA Bridge
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params += list(bridge.parameters())

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=0.01,
        )
        log.info("Оптимизатор: AdamW 8-bit (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=0.01,
        )
        log.info("Оптимизатор: AdamW (стандартный)")

    # ── 6. Training loop ────────────────────────────────────────────────────
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model.train()
    bridge.train()
    global_step = 0
    total_loss = 0.0

    log.info(f"Начинаем обучение: {cfg.num_epochs} эпох, "
             f"batch={cfg.batch_size}, grad_accum={cfg.grad_accum_steps}")
    log.info(f"Effective batch size = {cfg.batch_size * cfg.grad_accum_steps}")

    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(dataloader):
            if global_step >= cfg.max_steps:
                break

            # Перемещаем на устройство
            batch = {k: v.to(device) for k, v in batch.items()}

            # Устанавливаем маску для CIDA hook
            hook_mgr.set_attention_mask(batch.get("attention_mask"))
            hook_mgr.reset_step_counter()

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["input_ids"],  # Causal LM
            )

            loss = outputs.loss / cfg.grad_accum_steps
            loss.backward()

            total_loss += loss.item()

            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = total_loss / 10
                    gate_val = torch.sigmoid(bridge.gate).item()

                    # VRAM мониторинг
                    vram = ""
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1e9
                        reserved = torch.cuda.memory_reserved() / 1e9
                        vram = f" | VRAM: {allocated:.1f}/{reserved:.1f} GB"

                    log.info(
                        f"[Epoch {epoch+1}] Step {global_step}/{cfg.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"CIDA gate: {gate_val:.4f}{vram}"
                    )
                    total_loss = 0.0

        if global_step >= cfg.max_steps:
            break

    # ── 7. Cleanup ──────────────────────────────────────────────────────────
    hook_mgr.remove_hooks()
    log.info("=" * 60)
    log.info("  Обучение завершено!")
    log.info(f"  CIDA gate (финальный): {torch.sigmoid(bridge.gate).item():.4f}")
    log.info("=" * 60)

    return model, bridge


# ═════════════════════════════════════════════════════════════════════════════
# 5. DRY RUN UTILITIES (для тестирования без GPU)
# ═════════════════════════════════════════════════════════════════════════════

class _DummyConfig:
    hidden_size = 256
    use_return_dict = True

class _DummyOutput:
    def __init__(self, loss, hidden):
        self.loss = loss
        self.last_hidden_state = hidden

class _DummyModel(nn.Module):
    """Фейковая модель для dry_run тестов на CPU."""
    def __init__(self, d_model=256, num_layers=2):
        super().__init__()
        self.config = _DummyConfig()
        self.config.hidden_size = d_model
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=d_model*2,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, 1000)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        B, L = input_ids.shape
        x = torch.randn(B, L, self.config.hidden_size, device=input_ids.device)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, 1000),
            torch.randint(0, 1000, (B * L,), device=input_ids.device),
        )
        return _DummyOutput(loss, x)

    def parameters(self, recurse=True):
        return super().parameters(recurse)


class _DummyTokenizer:
    def __call__(self, texts, **kw):
        max_len = kw.get("max_length", 128)
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": [[1] * max_len for _ in texts],
            "attention_mask": [[1] * max_len for _ in texts],
        }


def _create_dummy_model(cfg):
    d_model = 256
    cfg.cida_d_hidden = 64  # Уменьшаем для CPU
    model = _DummyModel(d_model=d_model)
    tokenizer = _DummyTokenizer()
    return model, tokenizer


def _create_dummy_dataset(cfg):
    """Создаёт фейковый датасет для тестирования."""
    max_len = cfg.max_seq_length
    n = 32  # Маленький датасет

    class FakeDS:
        def __init__(self):
            self.data = {
                "input_ids": torch.randint(0, 1000, (n, max_len)),
                "attention_mask": torch.ones(n, max_len, dtype=torch.long),
            }
        def __len__(self):
            return n
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    return FakeDS()


# ═════════════════════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="CIDA Kairos: DiffusionGemma + Liquid Neural ODE"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Запуск с фейковой моделью на CPU (для тестов)"
    )
    parser.add_argument(
        "--model_id", type=str, default=None,
        help="HuggingFace model ID (default: DiffusionGemma)"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Максимум шагов обучения"
    )
    parser.add_argument(
        "--lora_r", type=int, default=None,
        help="LoRA rank"
    )
    parser.add_argument(
        "--cida_agents", type=int, default=None,
        help="Количество CIDA агентов"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = KairosConfig()

    # Применяем CLI overrides
    if args.model_id:
        cfg.model_id = args.model_id
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.lora_r:
        cfg.lora_r = args.lora_r
    if args.cida_agents:
        cfg.cida_num_agents = args.cida_agents

    if args.dry_run:
        cfg.max_steps = 20
        cfg.max_seq_length = 128
        cfg.batch_size = 2
        cfg.grad_accum_steps = 2

    train(cfg, dry_run=args.dry_run)
