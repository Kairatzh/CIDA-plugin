"""
CIDAPluginConfig — единая точка конфигурации для CIDA-Plugin.

v2: добавлены параметры для:
  - TransformerAgentUpdater (num_attn_heads)
  - EMA Reliability Tracker (rho_decay)
  - Learnable D-Schedule (learnable_d_schedule)
"""
from dataclasses import dataclass, asdict
import json
import os


@dataclass
class CIDAPluginConfig:
    """
    Конфигурация универсального CIDA-Plugin слоя.

    Параметры ядра
    --------------
    d_input : int
        Размерность pooled_output от внешнего энкодера.
        Примеры: 128 (bert-tiny), 512 (bert-medium), 768 (bert-base / distilbert).
    d_hidden : int
        Внутренняя размерность агентов плагина.
    d_message : int
        Размерность сообщений при коммуникации агентов.
    num_agents : int
        Количество агентов-спорщиков (рекомендуется 3–5).
    num_classes : int
        Число целевых классов.
    max_rounds : int
        Максимальное число раундов deliberation.

    Архитектурные параметры (v2)
    ----------------------------
    num_attn_heads : int
        Число голов в TransformerAgentUpdater (cross-attention).
        Должен делить d_hidden. По умолчанию 4.
    rho_decay : float
        Скорость затухания EMA для трекера надёжности агентов.
        0.9 = медленная адаптация, 0.5 = быстрая.
    learnable_d_schedule : bool
        Если True — расписание разногласий обучается как параметр
        в OmegaLossSystem (вместо жёстко заданных значений).

    Регуляризация
    -------------
    comm_dropout : float
        Dropout на канале коммуникации между агентами.
    early_stop_threshold : float or None
        Порог уверенности для досрочной остановки deliberation.
        Если None — всегда идёт max_rounds раундов.
    freeze_input_proj : bool
        Если True — входной проекционный слой не обучается.

    Ablation Flags
    --------------
    abl_no_pointers : bool      Отключить evidence pointers.
    abl_no_messages : bool      Отключить структурированные сообщения.
    abl_no_poe : bool           Отключить Product-of-Experts агрегацию.
    abl_no_communication : bool Полностью отключить коммуникацию.
    """
    # ─── Размерности ────────────────────────────────────────────────────────────
    d_input: int = 128
    d_hidden: int = 128
    d_message: int = 128

    # ─── Агенты и deliberation ───────────────────────────────────────────────────
    num_agents: int = 4
    num_classes: int = 2
    max_rounds: int = 2

    # ─── Архитектурные параметры v2 ──────────────────────────────────────────────
    num_attn_heads: int = 4         # для TransformerAgentUpdater
    rho_decay: float = 0.9          # для EMA Reliability Tracker
    learnable_d_schedule: bool = True  # обучаемое расписание разногласий

    # ─── Регуляризация ──────────────────────────────────────────────────────────
    comm_dropout: float = 0.3
    early_stop_threshold: float = 0.90

    # ─── Утилиты ────────────────────────────────────────────────────────────────
    freeze_input_proj: bool = False

    # ─── Ablation flags ─────────────────────────────────────────────────────────
    abl_no_pointers: bool = False
    abl_no_messages: bool = False
    abl_no_poe: bool = False
    abl_no_communication: bool = False

    def __post_init__(self):
        assert self.d_input > 0, "d_input должен быть положительным"
        assert self.num_agents >= 2, "Нужно минимум 2 агента для коммуникации"
        assert self.num_classes >= 2, "Нужно минимум 2 класса"
        assert self.max_rounds >= 1, "Нужен хотя бы 1 раунд deliberation"
        assert self.d_hidden % self.num_attn_heads == 0, (
            f"d_hidden ({self.d_hidden}) должен делиться на num_attn_heads ({self.num_attn_heads})"
        )
        assert 0.0 < self.rho_decay < 1.0, "rho_decay должен быть в (0, 1)"
        if self.early_stop_threshold is not None:
            assert 0.5 < self.early_stop_threshold <= 1.0, \
                "Порог early stopping должен быть в (0.5, 1.0]"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        import os
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        else:
            # Для HF Hub
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="config.json")
        
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
