"""run_all.py — Orchestrator: runs all pipeline scripts in correct order."""
from __future__ import annotations

import importlib
import time


STEPS = [
    "01_clean_prices",
    "02_clean_fundamentals",
    "03_clean_bloomberg",
    "04_clean_sectors",
    "05_feature_returns",
    "06_feature_fundamentals",
    "07_feature_indices",
    "08_assemble_x_ts",
    "09_assemble_x_static",
    "10_validate_final",
]


def main() -> None:
    print("=" * 60)
    print("  NeuralFactors Brasil — Data Pipeline")
    print("=" * 60)

    t0 = time.time()

    for step_name in STEPS:
        t_step = time.time()
        mod = importlib.import_module(step_name)
        mod.main()
        elapsed = time.time() - t_step
        print(f"      ({elapsed:.1f}s)\n")

    total = time.time() - t0
    print("=" * 60)
    print(f"  Pipeline complete in {total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
