# -*- coding: utf-8 -*-
"""
Объединяющий бенчмарк CIDA-Plugin.

Запускает оба эксперимента и сохраняет итоговый отчёт.

Использование:
    python experiments/run_plugin_benchmark.py
    python experiments/run_plugin_benchmark.py --dry-run
    python experiments/run_plugin_benchmark.py --only encoders
    python experiments/run_plugin_benchmark.py --only early_stop
"""
import sys
import os
import json
import argparse
import io
from datetime import datetime

# Принудительно UTF-8 вывод (Windows cp1251 fix)
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description="CIDA-Plugin Benchmark Suite")
    parser.add_argument("--dry-run", action="store_true", help="Быстрый smoke-test всех экспериментов")
    parser.add_argument("--only", choices=["encoders", "early_stop"], default=None,
                        help="Запустить только один эксперимент")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  CIDA-Plugin Benchmark Suite")
    print(f"  Запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Smoke-test: проверяем import ────────────────────────────────────────
    print("\n[0/2] Smoke-test: проверка import и forward pass...")
    try:
        import torch
        from cida_plugin import CIDAPlugin, CIDAPluginConfig

        cfg = CIDAPluginConfig(d_input=128, d_hidden=64, num_classes=2, max_rounds=2)
        plugin = CIDAPlugin(cfg)

        pooled = torch.randn(4, 128)
        seq_out = torch.randn(4, 32, 128)
        out = plugin(pooled, seq_output=seq_out)

        assert out["p_final"].shape == (4, 2), f"Неверная форма p_final: {out['p_final'].shape}"
        assert "rounds_used" in out, "Нет ключа rounds_used в выводе"
        print("  [OK] Import OK")
        print(f"  [OK] Forward pass OK | p_final: {out['p_final'].shape} | rounds: {out['rounds_used']}")
    except Exception as e:
        print(f"  [ERR] ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    all_results = {}

    # ── Эксперимент 1: Multi-Encoder ────────────────────────────────────────
    if args.only in (None, "encoders"):
        print("\n[1/2] Запуск: exp_plugin_encoders...")
        try:
            from experiments.exp_plugin_encoders import run_experiment as run_encoders
            res = run_encoders(dry_run=args.dry_run)
            all_results["multi_encoder"] = res
        except Exception as e:
            print(f"  ОШИБКА в exp_plugin_encoders: {e}")
            import traceback
            traceback.print_exc()

    # ── Эксперимент 2: Early Stopping ───────────────────────────────────────
    if args.only in (None, "early_stop"):
        print("\n[2/2] Запуск: exp_plugin_early_stop...")
        try:
            from experiments.exp_plugin_early_stop import run_experiment as run_early_stop
            res = run_early_stop(dry_run=args.dry_run)
            all_results["early_stop"] = res
        except Exception as e:
            print(f"  ОШИБКА в exp_plugin_early_stop: {e}")
            import traceback
            traceback.print_exc()

    # ── Итоговый отчёт ──────────────────────────────────────────────────────
    out_path = os.path.join(ROOT, "results", "plugin_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": args.dry_run,
        "results": all_results,
    }

    # Безопасная сериализация (rounds_distribution_pct keys)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(i) for i in obj]
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(report), f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  Бенчмарк завершён!")
    print(f"  Итоговый отчёт → {out_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
