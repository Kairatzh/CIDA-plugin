"""
benchmark_golden_eval.py — Official Scientific Benchmark Suite Entrypoint

Redirects directly to benchmark_scientific package with 100% empirical evaluation,
zero synthetic simulation, 5-paradigm comparison, sandboxed unit test execution,
and publication-grade charts.
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from benchmark_scientific.run_benchmark import main

if __name__ == "__main__":
    main()
