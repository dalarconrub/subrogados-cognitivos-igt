"""Anexo A, apartado A.5 · Análisis estadísticos: módulos comunes y scripts.

Parche que todos los scripts del apartado aplican antes de calcular: elimina de las particiones las 46
copias repetidas de la segunda sesión de Sullivan-Toole et al. (2022), ya presente en el corpus.
"""

from __future__ import annotations

RETEST_EXPS = {
    "igt_sullivantoole2022_retest_vv1", "igt_sullivantoole2022_retest_vv2",
    "igt_sullivantoole2022_retest_vv3", "igt_sullivantoole2022_retest_vv4",
}
N_RETEST = 46


def retest_ids(manifest) -> set[str]:
    return {r["subject_id"] for r in manifest if r["experiment"] in RETEST_EXPS}


def patch_build_partitions(base_module, enabled: bool = True):
    """Envuelve build_partitions para retirar las 46 filas retest de C.

    Con enabled=False deja el comportamiento canon (puerta de validación)."""
    orig = base_module.build_partitions

    def build_partitions_dedup(manifest):
        parts = orig(manifest)
        if not enabled:
            return parts
        ids = retest_ids(manifest)
        antes = len(parts["C_EXTERNAL_510"])
        parts["C_EXTERNAL_510"] = [s for s in parts["C_EXTERNAL_510"] if s not in ids]
        assert antes - len(parts["C_EXTERNAL_510"]) == N_RETEST, (antes, len(parts["C_EXTERNAL_510"]))
        return parts

    base_module.build_partitions = build_partitions_dedup
    return base_module
