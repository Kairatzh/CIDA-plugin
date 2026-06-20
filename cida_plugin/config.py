"""
CIDAPluginConfig — единая точка конфигурации для CIDA-Plugin v3.
"""
from dataclasses import dataclass, asdict
import json
import os


@dataclass
class CIDAPluginConfig:
    """
    Конфигурация универсального CIDA-Plugin слоя (v3 — упрощённая архитектура).

    Параметры ядра
    --------------
    d_input : int
        Размерность pooled_output от внешнего энкодера.
        Примеры: 128 (bert-tiny), 768 (bert-base / distilbert), 1024 (DenseNet121).
    d_hidden : int
        Внутренняя размерность агентов плагина.
    d_message : int
        Размерность сообщений при коммуникации агентов.
    num_agents : int
        Количество агентов (рекомендуется 4: Prosecutor, Defender, Skeptic, Integrator).
    num_classes : int
        Число целевых классов.
    max_rounds : int
        Максимальное число раундов deliberation.
    multi_label : bool
        True для multi-label задач (BCE), False для single-label (CE).

    Архитектурные параметры
    -----------------------
    num_attn_heads : int
        Число голов в TransformerAgentUpdater (cross-attention).
        Должен делить d_hidden.

    Role Priors
    -----------
    role_prior_blend : float
        Сила смешивания роли с данными (0.0 = без priors, 1.0 = только priors).
        Рекомендуется 0.30 (30% prior, 70% данные).
    positive_class_idx : int
        Индекс "положительного" класса для multi-class задач (K>2).
        По умолчанию 1, но может быть 0, 2 и т.д. в зависимости от датасета.

    Loss гиперпараметры
    -------------------
    lambda_cal : float
        Вес calibration loss (Brier score). Рекомендуется 0.3–0.5.
    lambda_ac : float
        Вес anti-collapse loss. Рекомендуется 0.1–0.3.
    min_disagreement : float
        Минимальный порог несогласия для anti-collapse.

    Регуляризация
    -------------
    comm_dropout : float
        Dropout на канале коммуникации между агентами.
    early_stop_threshold : float or None
        Порог уверенности для досрочной остановки deliberation (inference only).
        Если None — всегда идёт max_rounds раундов.
    freeze_input_proj : bool
        Если True — входной проекционный слой не обучается.

    Ablation Flags
    --------------
    abl_no_pointers : bool
        Отключить evidence pointers.
    abl_no_messages : bool
        Отключить формулировку сообщений.
    abl_no_communication : bool
        Полностью отключить коммуникацию.

    Perspective Projector
    ---------------------
    use_perspective_projector : bool
        Если True — каждый агент имеет свою проекцию входа (per-agent view).
        Если False — все агенты начинают с одного представления (expand).
        По умолчанию False для упрощённой архитектуры v3.

    Test-Time Training (TTT)
    -------------------------
    use_ttt : bool
        Агенты адаптируют свои веса к каждому входу перед deliberation.
    ttt_steps : int
        Число шагов внутренней оптимизации (K).
    ttt_lr : float
        Learning rate для inner Adam.
    ttt_mask_ratio : float
        Доля маскируемых компонент скрытого состояния.

    Liquid Neural ODE Dynamics
    --------------------------
    use_liquid_dynamics : bool
        Заменяет дискретные раунды на непрерывное ODE: ds/dt = -s/τ + F(s,r,e).
    liquid_solver : str
        Метод интегрирования: 'euler', 'dopri5' и др.
    liquid_atol : float
        Абсолютная толерантность для адаптивных решателей.
    liquid_rtol : float
        Относительная толерантность для адаптивных решателей.
    trajectory_save_every : int
        Сохранять каждое N-е состояние в trajectory для ODE решателя.
        Для адаптивных решателей (dopri5) рекомендуется 10-50 для экономии памяти.
        Для euler можно использовать 1.
    """
    # ─── Размерности ────────────────────────────────────────────────────────────
    d_input: int = 128
    d_hidden: int = 128
    d_message: int = 128

    # ─── Агенты и deliberation ───────────────────────────────────────────────────
    num_agents: int = 4
    num_classes: int = 2
    max_rounds: int = 3
    multi_label: bool = False

    # ─── Архитектурные параметры ─────────────────────────────────────────────────
    num_attn_heads: int = 4

    # ─── Role Priors ─────────────────────────────────────────────────────────────
    role_prior_blend: float = 0.30
    positive_class_idx: int = 1

    # ─── Loss гиперпараметры ─────────────────────────────────────────────────────
    lambda_cal: float = 0.4
    lambda_ac: float = 0.2
    min_disagreement: float = 0.08

    # ─── Регуляризация ──────────────────────────────────────────────────────────
    comm_dropout: float = 0.2
    early_stop_threshold: float = 0.90
    freeze_input_proj: bool = False

    # ─── Ablation flags ─────────────────────────────────────────────────────────
    abl_no_pointers: bool = False
    abl_no_messages: bool = False
    abl_no_communication: bool = False

    # ─── Perspective Projector ─────────────────────────────────────────────────
    use_perspective_projector: bool = True

    # ─── Test-Time Training ──────────────────────────────────────────────────────
    use_ttt: bool = False
    ttt_steps: int = 3
    ttt_lr: float = 1e-3
    ttt_mask_ratio: float = 0.15

    # ─── Liquid Neural ODE Dynamics ──────────────────────────────────────────────
    use_liquid_dynamics: bool = False
    liquid_solver: str = "euler"
    liquid_atol: float = 1e-3
    liquid_rtol: float = 1e-3
    trajectory_save_every: int = 10

    # ─── CIDA v4: Architectural Improvements ─────────────────────────────────────
    # Disagreement-Routed Sparse Communication (Li et al. EMNLP 2024)
    # Каждый агент слушает только top-K по disagreement, не всех.
    sparse_comm_k: int = 2

    # Anonymous Message Passing (Choi et al. ACL 2025)
    # Убирает s (identity) из сообщений → агент слышит аргумент, а не авторитет.
    anonymous_messages: bool = True

    # Dynamic Trust Weights (CortexDebate MDM, ACL 2025)
    # EMA trust вместо статических ROLE_WEIGHTS.
    use_dynamic_trust: bool = False
    trust_ema_gamma: float = 0.9

    def __post_init__(self):
        assert self.d_input > 0, "d_input must be positive"
        assert self.num_agents >= 2, "Need at least 2 agents"
        assert self.num_classes >= 2, "Need at least 2 classes"
        assert self.max_rounds >= 1, "Need at least 1 deliberation round"
        assert self.d_hidden % self.num_attn_heads == 0, (
            f"d_hidden ({self.d_hidden}) must be divisible by "
            f"num_attn_heads ({self.num_attn_heads})"
        )
        assert 0.0 <= self.role_prior_blend <= 1.0, (
            "role_prior_blend must be in [0.0, 1.0]"
        )
        if self.early_stop_threshold is not None:
            assert 0.5 < self.early_stop_threshold <= 1.0, (
                "early_stop_threshold must be in (0.5, 1.0]"
            )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        # Filter out unknown keys for forward compatibility
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(
                pretrained_model_name_or_path, "config.json"
            )
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="config.json",
            )

        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
