#!/usr/bin/env python3
"""Anexo A, apartado A.4 · El ciclo completo: entrenar el adaptador y evaluar ensayo a ensayo.

Biblioteca que usan las evaluaciones para el contraste pareado por sesión entre dos modelos: diferencia
de verosimilitud por sesión, t pareada unilateral, d de Cohen con signo e intervalo por bootstrap con la semilla
del estudio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

try:
    import numpy as np
    from scipy import stats as scstats
except ImportError:
    print("ERROR: numpy + scipy required. pip install numpy scipy", file=sys.stderr)
    sys.exit(3)

DEFAULT_SEED = 20260506
DEFAULT_N_BOOT = 10_000


def cohens_d_paired(deltas: Sequence[float]) -> float:
    """Cohen's d for paired differences (one-sample d against zero)."""
    if not deltas:
        return float("nan")
    arr = np.asarray(deltas, dtype=float)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()) / sd if sd > 0 else float("nan")


def bootstrap_ci_mean(deltas: Sequence[float], n_boot: int = DEFAULT_N_BOOT,
                     seed: int = DEFAULT_SEED, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap CI for mean of paired deltas."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[b] = arr[idx].mean()
    return float(np.quantile(boot_means, alpha / 2)), float(np.quantile(boot_means, 1 - alpha / 2))


def paired_t_test_one_sided(deltas: Sequence[float]) -> tuple[float, float, int]:
    """Return (t_stat, p_one_sided, df). One-sided test that mean < 0."""
    arr = np.asarray(deltas, dtype=float)
    t_stat, p_two = scstats.ttest_1samp(arr, 0.0)
    p_one = (p_two / 2) if t_stat < 0 else 1.0 - (p_two / 2)
    return float(t_stat), float(p_one), len(arr) - 1


def paired_t_test_bootstrap(deltas: Sequence[float], seed: int = DEFAULT_SEED,
                            n_boot: int = DEFAULT_N_BOOT) -> dict:
    """All-in-one: paired t-test + bootstrap CI + Cohen's d."""
    if not deltas:
        raise ValueError("Empty deltas")
    arr = np.asarray(deltas, dtype=float)
    t_stat, p_one, df = paired_t_test_one_sided(deltas)
    ci_lo, ci_hi = bootstrap_ci_mean(deltas, n_boot=n_boot, seed=seed)
    return {
        "n": len(arr),
        "delta_mean": float(arr.mean()),
        "delta_sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ci_lo_95": ci_lo,
        "ci_hi_95": ci_hi,
        "t_stat": t_stat,
        "df": df,
        "p_value_one_sided": p_one,
        "cohens_d": cohens_d_paired(deltas),
        "support_frac_negative": float((arr < 0).mean()),
        "n_bootstrap": n_boot,
        "seed": seed,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deltas", type=Path, required=True,
                   help="JSON array of paired deltas (Centaur NLL - baseline NLL)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    deltas = json.loads(args.deltas.read_text(encoding="utf-8"))
    if not isinstance(deltas, list) or not all(isinstance(x, (int, float)) for x in deltas):
        print("ERROR: --deltas must be a JSON array of numbers", file=sys.stderr)
        return 1

    stats = paired_t_test_bootstrap(deltas, seed=args.seed, n_boot=args.n_boot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"OK: stats written to {args.output}")
    print(f"  n = {stats['n']}")
    print(f"  Δ mean = {stats['delta_mean']:+.4f}")
    print(f"  CI 95% = [{stats['ci_lo_95']:+.4f}, {stats['ci_hi_95']:+.4f}]")
    print(f"  t({stats['df']}) = {stats['t_stat']:.2f}, P (one-sided) = {stats['p_value_one_sided']:.2e}")
    print(f"  Cohen's d = {stats['cohens_d']:+.3f}")
    print(f"  Support frac (Δ<0) = {stats['support_frac_negative']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
