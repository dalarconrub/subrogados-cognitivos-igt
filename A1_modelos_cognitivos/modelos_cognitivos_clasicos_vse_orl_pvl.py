"""Anexo A, apartado A.1 · Modelos cognitivos clásicos: parametrización y ajuste.

Implementa los tres modelos cognitivos clásicos (regla de aprendizaje, función de valoración y regla
de decisión softmax de cada uno) y calcula la verosimilitud de cada elección humana ensayo a ensayo. Cada
parámetro se estima sin restricciones y una transformación fija lo lleva a su rango: sigmoide para las tasas y la
sensibilidad, tangente hiperbólica escalada para la bonificación de exploración de VSE, softplus para la
consistencia de VSE y el decaimiento perseverativo de ORL, sigmoide escalada para los parámetros de PVL-Δ y ninguna
transformación para los dos pesos de ORL. El ajuste es por máxima verosimilitud con un único vector de parámetros común a todas las sesiones,
validación cruzada de cinco pliegues formados dentro de cada grupo de procedencia, optimizador L-BFGS-B con 32
arranques aleatorios y 1.000 iteraciones por arranque, y semilla 20260506.
"""


from __future__ import annotations

import argparse
import hashlib
import json
import math
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import optimize

DECK_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


# ---------------------------------------------------------------------------
# VSE — Value-plus-Sequential-Exploration (Ligneul 2019)
# ---------------------------------------------------------------------------
def vse_per_trial_nll(params: np.ndarray, trials: list[dict[str, Any]]) -> np.ndarray:
    """Return per-trial NLL = -log p(observed deck) under VSE with given parameters.

    Parameter transforms (Ligneul priors): all 5 transformed from unbounded space:
      theta = sigmoid(p[0])              ∈ (0, 1)
      delta = sigmoid(p[1])              ∈ (0, 1)
      alpha = sigmoid(p[2])              ∈ (0, 1)
      phi   = 10 * tanh(p[3])            ∈ (-10, 10)
      consistency = 3 ** softplus(p[4]) - 1   (positive, bounded below by 0)
    """
    theta = _sigmoid(params[0])
    delta = _sigmoid(params[1])
    alpha = _sigmoid(params[2])
    phi = 10.0 * np.tanh(params[3])
    consistency = 3.0 ** _softplus(params[4]) - 1.0

    exploit = np.zeros(4)
    explore = np.full(4, phi)

    nlls = np.empty(len(trials))
    for i, t in enumerate(trials):
        d = DECK_TO_INDEX.get(t["deck"], -1)
        if d < 0:
            nlls[i] = np.nan
            continue
        # Decision
        combined = exploit + explore
        logits = combined * consistency
        # log-softmax
        log_z = _logsumexp(logits)
        logp = logits[d] - log_z
        nlls[i] = -float(logp)
        # State update (Ligneul evolution)
        gain = float(t.get("gain", 0.0))
        loss = float(t.get("loss", 0.0))
        new_exploit = np.empty(4)
        new_explore = np.empty(4)
        for de in range(4):
            if de != d:
                new_exploit[de] = exploit[de] * delta
                new_explore[de] = explore[de] + alpha * (phi - explore[de])
            else:
                new_exploit[de] = exploit[de] * delta + (abs(gain) ** theta) - (abs(loss) ** theta)
                new_explore[de] = 0.0
        exploit = new_exploit
        explore = new_explore
    return nlls


def vse_total_nll(params: np.ndarray, trials: list[dict[str, Any]]) -> float:
    nlls = vse_per_trial_nll(params, trials)
    finite = nlls[np.isfinite(nlls)]
    return float(finite.sum()) if finite.size > 0 else float("inf")


# ---------------------------------------------------------------------------
# ORL — Outcome-Representation Learning (Haines 2018)
# ---------------------------------------------------------------------------
def orl_per_trial_nll(params: np.ndarray, trials: list[dict[str, Any]]) -> np.ndarray:
    """ORL per-trial NLL.

    Parameter transforms (matches Sullivan-Toole Stan priors):
      Arew  = sigmoid(p[0])           ∈ (0, 1)
      Apun  = sigmoid(p[1])           ∈ (0, 1)
      K     = 3 ** softplus(p[2]) - 1 ∈ (0, ∞)  — decay strength
      betaF = p[3]                    unbounded
      betaP = p[4]                    unbounded
    """
    Arew = _sigmoid(params[0])
    Apun = _sigmoid(params[1])
    K = 3.0 ** _softplus(params[2]) - 1.0
    betaF = float(params[3])
    betaP = float(params[4])

    EV = np.zeros(4)
    EF = np.zeros(4)  # expected frequency of wins
    PS = np.zeros(4)  # perseverance

    nlls = np.empty(len(trials))
    for i, t in enumerate(trials):
        d = DECK_TO_INDEX.get(t["deck"], -1)
        if d < 0:
            nlls[i] = np.nan
            continue
        logits = EV + EF * betaF + PS * betaP
        log_z = _logsumexp(logits)
        logp = logits[d] - log_z
        nlls[i] = -float(logp)
        # Updates
        gain = float(t.get("gain", 0.0))
        loss = float(t.get("loss", 0.0))
        net = gain - loss  # outcome neto: 'loss' es magnitud POSITIVA en el manifest (gain>=0, loss>=0) -> se resta
        sign_out = 1.0 if net > 0 else (-1.0 if net < 0 else 0.0)
        # EV: chosen deck moves toward outcome
        if net >= 0:
            EV[d] += Arew * (net - EV[d])
        else:
            EV[d] += Apun * (net - EV[d])
        # EF: chosen deck moves toward sign_out, unchosen decks shift opposite
        for de in range(4):
            if de == d:
                if net >= 0:
                    EF[de] += Arew * (sign_out - EF[de])
                else:
                    EF[de] += Apun * (sign_out - EF[de])
            else:
                if net >= 0:
                    EF[de] += Apun * (-sign_out / 3.0 - EF[de])
                else:
                    EF[de] += Arew * (-sign_out / 3.0 - EF[de])
        # PS: perseverance decays toward 0 by 1/(1+K); chosen deck gets boost
        PS = PS / (1.0 + K)
        PS[d] = 1.0
    return nlls


def orl_total_nll(params: np.ndarray, trials: list[dict[str, Any]]) -> float:
    nlls = orl_per_trial_nll(params, trials)
    finite = nlls[np.isfinite(nlls)]
    return float(finite.sum()) if finite.size > 0 else float("inf")


# ---------------------------------------------------------------------------
# PVL-Δ — Prospect Valence Learning with delta-rule (Ahn 2008; Steingroever 2018)
# ---------------------------------------------------------------------------
def pvldelta_per_trial_nll(params: np.ndarray, trials: list[dict[str, Any]]) -> np.ndarray:
    """PVL-Δ per-trial NLL.

    Parameter transforms (matches Ahn 2008 / Steingroever 2018 priors):
      A     = sigmoid(p[0])              ∈ (0, 1)   — recency/learning rate (delta)
      alpha = 2.0 * sigmoid(p[1])        ∈ (0, 2)   — utility curvature
      lam   = 10.0 * sigmoid(p[2])       ∈ (0, 10)  — loss aversion (lambda)
      c     = 5.0 * sigmoid(p[3])        ∈ (0, 5)   — softmax consistency

    Utility: u(net) = sign(net) * |net|^alpha for gains; -lam * |net|^alpha for losses.
    Update:  EV(d_chosen) += A * (u - EV(d_chosen))
    Choice:  softmax(c * EV)
    """
    A = _sigmoid(params[0])
    alpha = 2.0 * _sigmoid(params[1])
    lam = 10.0 * _sigmoid(params[2])
    c = 5.0 * _sigmoid(params[3])

    EV = np.zeros(4)

    nlls = np.empty(len(trials))
    for i, t in enumerate(trials):
        d = DECK_TO_INDEX.get(t["deck"], -1)
        if d < 0:
            nlls[i] = np.nan
            continue
        logits = c * EV
        log_z = _logsumexp(logits)
        logp = logits[d] - log_z
        nlls[i] = -float(logp)
        # PVL utility
        gain = float(t.get("gain", 0.0))
        loss = float(t.get("loss", 0.0))
        net = gain - loss  # outcome neto: 'loss' es magnitud POSITIVA en el manifest (gain>=0, loss>=0) -> se resta
        if net >= 0:
            u = (net ** alpha) if net > 0 else 0.0
        else:
            u = -lam * (abs(net) ** alpha)
        # Delta-rule update
        EV[d] += A * (u - EV[d])
    return nlls


def pvldelta_total_nll(params: np.ndarray, trials: list[dict[str, Any]]) -> float:
    nlls = pvldelta_per_trial_nll(params, trials)
    finite = nlls[np.isfinite(nlls)]
    return float(finite.sum()) if finite.size > 0 else float("inf")


# ---------------------------------------------------------------------------
# MLE fitter (multi-start L-BFGS-B)
# ---------------------------------------------------------------------------
def _multistart(obj, n_params: int, n_starts: int, seed: int, maxiter: int) -> dict[str, Any]:
    """Multi-start L-BFGS-B over an objective obj(params)->float. Returns best fit +
    convergence metadata (P1.2: callers must gate status on `success`+finite `fun`)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        x0 = rng.normal(0, 1, size=n_params)
        try:
            res = optimize.minimize(obj, x0, method="L-BFGS-B",
                                    options={"maxiter": maxiter, "ftol": 1e-6})
        except Exception:  # pragma: no cover
            continue
        if best is None or (res.fun < best.fun and np.isfinite(res.fun)):
            best = res
    return {
        "params_raw": best.x.tolist() if best is not None else None,
        "fun": float(best.fun) if best is not None else None,
        "success": bool(best.success) if best is not None else False,
        "nit": int(best.nit) if best is not None else 0,
        "message": str(best.message) if best is not None else "no successful start",
        "n_starts": n_starts,
        "maxiter": maxiter,
        "seed": seed,
    }


def fit_mle(
    nll_fn,
    trials: list[dict[str, Any]],
    n_params: int = 5,
    n_starts: int = 8,
    seed: int = 20260513,
    maxiter: int = 200,
) -> dict[str, Any]:
    """Per-subject MLE: minimize total NLL over one subject's trials."""
    return _multistart(lambda p: nll_fn(p, trials), n_params, n_starts, seed, maxiter)


def fit_mle_pooled(
    total_fn,
    trials_list: list[list[dict[str, Any]]],
    n_params: int = 5,
    n_starts: int = 8,
    seed: int = 20260506,
    maxiter: int = 200,
) -> dict[str, Any]:
    """Pooled MLE (complete pooling, à la Binz 2025a): minimize the SUM of total NLL
    over many subjects' full sessions -> one shared parameter vector."""
    def obj(p):
        s = 0.0
        for tr in trials_list:
            v = total_fn(p, tr)
            if not np.isfinite(v):
                return float("inf")
            s += v
        return s
    return _multistart(obj, n_params, n_starts, seed, maxiter)


def _model_spec(model: str):
    if model == "vse":
        return vse_per_trial_nll, vse_total_nll, 5, ["theta_raw", "delta_raw", "alpha_raw", "phi_raw", "beta_raw"]
    if model == "orl":
        return orl_per_trial_nll, orl_total_nll, 5, ["Arew_raw", "Apun_raw", "K_raw", "betaF", "betaP"]
    if model == "pvldelta":
        return pvldelta_per_trial_nll, pvldelta_total_nll, 4, ["A_raw", "alpha_raw", "lambda_raw", "c_raw"]
    raise ValueError(f"unknown model: {model}")


def _prov(subj: dict[str, Any]) -> dict[str, Any]:
    """Row provenance (P1.3): never reconstruct experiment/cohort from subject_id strings."""
    return {
        "subject_id": subj["subject_id"],
        "manifest": subj.get("manifest"),
        "experiment": subj.get("experiment"),
        "cohort": subj.get("cohort"),
        "igt_version": subj.get("igt_version"),
        "deck_map": subj.get("deck_map"),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x))) if abs(x) < 50 else (1.0 if x > 0 else 0.0)


def _softplus(x: float) -> float:
    x = float(x)
    return math.log1p(math.exp(x)) if x < 50 else x


def _logsumexp(arr: np.ndarray) -> float:
    m = float(np.max(arr))
    return m + float(np.log(np.exp(arr - m).sum()))


def fit_dataset(
    model: str,
    manifest_path: Path,
    out_path: Path,
    n_starts: int = 8,
    fit_trials: int | None = None,
    fit_frac: float | None = None,
    min_eval_trials: int = 1,
    min_fit_trials: int = 5,
    maxiter: int = 200,
) -> None:
    """Fit a cognitive baseline per subject and write per-trial NLL traces.

    Two modes:

    - **In-sample full-session** (default; ``fit_trials``/``fit_frac`` both None):
      fit on all trials and report NLL on the same trials. This is a robustness /
      descriptive result ("which model best DESCRIBES IGT data"), NOT a predictive
      metric — it is optimistic (fit == eval) and the optimism grows with #params,
      so it is not a fair Centaur contrast.

    - **Subject-adaptive temporal holdout** (``fit_trials=k`` or ``fit_frac=f``):
      fit ONLY on the first k trials, evaluate held-out NLL on the remaining
      trials[k:]. This is the H1 OUT-OF-SAMPLE metric (`metric=
      "subject_adaptive_temporal_holdout"`), comparable to Centaur once Centaur is
      re-scored on the same held-out suffix. The window policy (k / f) IS the H1
      estimand — fixed by the study design, not hard-coded here.

      NOTE (sequential models): the eval NLL is computed by running the model
      forward over the FULL sequence with the fitted params and slicing [k:], so
      the latent state (EV / exploit-explore / PS) carries from the fit window into
      the eval window. Re-running on trials[k:] from scratch would be wrong (it
      would reset the state and treat trial k as trial 0). The fit optimizer only
      ever sees trials[:k], so held-out trials never leak into fitting.
    """
    per_trial_fn, total_fn, n_params, param_names = _model_spec(model)

    holdout = fit_trials is not None or fit_frac is not None
    metric = "subject_adaptive_temporal_holdout" if holdout else "in_sample_full_session"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_excluded = 0
    n_total = sum(1 for _ in manifest_path.open())
    with manifest_path.open() as inp, out_path.open("w") as outp:
        for line in inp:
            subj = json.loads(line)
            trials = subj["trials"]
            if not trials:
                continue
            n = len(trials)
            base = {
                "model": model,
                "metric": metric,
                **_prov(subj),
                "n_trials": n,
            }

            if not holdout:
                # In-sample full-session (unchanged math; just tagged with `metric`).
                mle = fit_mle(total_fn, trials, n_params=n_params, n_starts=n_starts, maxiter=maxiter)
                params = np.array(mle["params_raw"]) if mle["params_raw"] else None
                per_trial = per_trial_fn(params, trials).tolist() if params is not None else None
                outp.write(json.dumps({
                    **base,
                    "mle": mle,
                    "param_names": param_names,
                    "per_trial_nll": per_trial,
                }) + "\n")
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  [{n_done}/{n_total}] {subj['subject_id']}")
                continue

            # --- Temporal-holdout mode ---
            k = int(fit_trials) if fit_trials is not None else int(math.floor(fit_frac * n))
            n_eval = n - k
            if k < min_fit_trials or n_eval < min_eval_trials:
                outp.write(json.dumps({
                    **base, "fit_trials": k, "n_fit": k, "n_eval": n_eval,
                    "status": "excluded_window_too_small",
                }) + "\n")
                n_excluded += 1
                continue

            mle = fit_mle(total_fn, trials[:k], n_params=n_params, n_starts=n_starts, maxiter=maxiter)
            params = np.array(mle["params_raw"]) if mle["params_raw"] else None
            # P1.2: only a converged, finite fit is admissible to the metric.
            if params is None or not mle["success"] or not np.isfinite(mle["fun"]):
                outp.write(json.dumps({
                    **base, "fit_trials": k, "n_fit": k, "n_eval": n_eval,
                    "mle": mle,
                    "status": "fit_failed" if params is None else "fit_not_converged",
                }) + "\n")
                n_excluded += 1
                continue

            # Forward over the full sequence; slice fit/eval windows (state carries over).
            per_trial_all = per_trial_fn(params, trials)
            fit_nll = per_trial_all[:k]
            eval_nll = per_trial_all[k:]
            eval_finite = eval_nll[np.isfinite(eval_nll)]
            fit_finite = fit_nll[np.isfinite(fit_nll)]
            status = "ok" if eval_finite.size else "eval_nonfinite"
            outp.write(json.dumps({
                **base,
                "fit_trials": k,
                "n_fit": k,
                "n_eval": n_eval,
                "eval_trial_start": k,  # held-out window = trials[k:]; pair Centaur per-trial on same indices
                "fit_policy": {"type": "trials" if fit_trials is not None else "frac",
                               "value": fit_trials if fit_trials is not None else fit_frac},
                "mle": mle,             # mle["fun"] = fit-window total NLL
                "param_names": param_names,
                "eval_mean_nll": float(eval_finite.mean()) if eval_finite.size else None,
                "eval_sum_nll": float(eval_finite.sum()) if eval_finite.size else None,
                "fit_mean_nll": float(fit_finite.mean()) if fit_finite.size else None,
                "per_trial_nll": eval_nll.tolist(),       # held-out per-trial NLL = the H1 quantity
                "fit_per_trial_nll": fit_nll.tolist(),
                "status": status,
            }) + "\n")
            if status == "ok":
                n_done += 1
            else:
                n_excluded += 1
            if n_done % 25 == 0:
                print(f"  [{n_done}/{n_total}] {subj['subject_id']}")
    extra = f" ({n_excluded} excluded)" if n_excluded else ""
    print(f"✓ fit {n_done} subjects → {out_path}{extra}  [metric={metric}]")


def _process_fold(payload):
    """Worker (top-level → picklable): ONE (experiment, fold) of ONE model. Returns
    (exp, f_i, rows, n_ok, n_excl). The fold split is recomputed deterministically
    (rng seeded by [seed, crc32(exp)]) and fold `f_i` is selected, so the output for a
    given (exp, f_i) is IDENTICAL whether run serially or in parallel, and identical to
    the previous per-experiment worker. Granularity = (experiment, fold) so the slowest
    experiment's K folds spread across cores ([revisión interna]: parallelize folds, not lower n_starts)."""
    model, exp, subs, kfolds, seed, n_starts, maxiter, min_train_subjects, f_i = payload
    per_trial_fn, total_fn, n_params, param_names = _model_spec(model)
    metric = "pooled_ml_subject_kfold"
    n = len(subs)
    k_eff = min(kfolds, n)
    tiny = n < 25
    exp_crc = int(zlib.crc32(exp.encode()))
    folds = np.array_split(np.random.default_rng([seed, exp_crc]).permutation(n), k_eff)
    fold = folds[f_i]
    fold_policy = {"unit": "subject", "within": "experiment", "k": k_eff,
                   "seed": seed, "exp_crc32": exp_crc}  # E4: record exp hash
    rows = []
    n_ok = n_excl = 0
    eval_set = {int(x) for x in fold}
    train_subs = [subs[i] for i in range(n) if i not in eval_set]
    eval_subs = [subs[i] for i in range(n) if i in eval_set]
    n_train_trials = sum(len(s["trials"]) for s in train_subs)
    # P3: fold membership hashes (reproducibility without re-running the splitter).
    train_sha = hashlib.sha256("\n".join(sorted(s["subject_id"] for s in train_subs)).encode()).hexdigest()
    eval_sha = hashlib.sha256("\n".join(sorted(s["subject_id"] for s in eval_subs)).encode()).hexdigest()
    low_train = len(train_subs) < max(8, 2 * n_params)  # E3: flag (not exclude)

    def _row(subj, **extra):
        return {"model": model, "metric": metric, **_prov(subj),
                "n_trials": len(subj["trials"]), "fold": f_i,
                "fold_policy": fold_policy, "tiny_cohort": tiny,
                "train_subject_ids_sha256": train_sha,
                "eval_fold_subject_ids_sha256": eval_sha,
                "low_train_warning": low_train, **extra}

    if len(train_subs) < min_train_subjects:
        for s in eval_subs:
            rows.append(_row(s, n_train_subjects=len(train_subs), status="excluded_tiny_train"))
            n_excl += 1
        return exp, f_i, rows, n_ok, n_excl

    mle = fit_mle_pooled(total_fn, [s["trials"] for s in train_subs],
                         n_params=n_params, n_starts=n_starts, seed=seed, maxiter=maxiter)
    params = np.array(mle["params_raw"]) if mle["params_raw"] else None
    ok_fit = params is not None and mle["success"] and np.isfinite(mle["fun"])
    for s in eval_subs:
        if not ok_fit:  # P1.2: do not admit non-converged pooled fits
            rows.append(_row(s, n_train_subjects=len(train_subs), n_train_trials=n_train_trials,
                             mle=mle, status="fit_failed" if params is None else "fit_not_converged"))
            n_excl += 1
            continue
        pt = per_trial_fn(params, s["trials"])
        fin = pt[np.isfinite(pt)]
        status = "ok" if fin.size else "eval_nonfinite"
        rows.append(_row(s, n_train_subjects=len(train_subs), n_train_trials=n_train_trials,
                         mle=mle, param_names=param_names,
                         eval_mean_nll=float(fin.mean()) if fin.size else None,
                         eval_sum_nll=float(fin.sum()) if fin.size else None,
                         per_trial_nll=pt.tolist(), status=status))
        n_ok += 1 if status == "ok" else 0
        n_excl += 0 if status == "ok" else 1
    return exp, f_i, rows, n_ok, n_excl


def fit_dataset_pooled_kfold(
    model: str,
    manifest_path: Path,
    out_path: Path,
    kfolds: int = 5,
    seed: int = 20260506,
    n_starts: int = 8,
    maxiter: int = 200,
    min_train_subjects: int = 2,
    jobs: int = 1,
) -> None:
    """H1 PRIMARY metric — pooled-ML held-out participants (faithful Binz 2025a protocol).

    For each `experiment`, split SUBJECTS into K deterministic folds. For each fold:
    fit ONE shared parameter vector by maximum likelihood on the *pooled* full sessions
    of the training subjects (the other folds), then evaluate the held-out subjects'
    full sessions with that shared vector. Every subject ends up with exactly one
    out-of-sample cognitive NLL (`metric="pooled_ml_subject_kfold"`), pairable against
    Centaur's session-level `nll_centaur.json`. Complete pooling → no subject-level
    importance weights → no PSIS/C1 degeneracy. Tiny experiments (n<25) use
    K=min(kfolds, n) + `tiny_cohort:true`; folds with train < min_train_subjects excluded.

    `jobs>1` parallelizes ACROSS **(experiment, fold)** tasks ([revisión interna]) via
    ProcessPoolExecutor — so the slowest experiment's K folds spread across cores instead
    of pinning to one. Output is identical to serial (each (exp,fold) seeded by [seed,
    crc32(exp)]). **Resumable per (experiment, fold)**: pairs already present in `out_path`
    are skipped (results written via `as_completed` as soon as each fold finishes, so a
    studio restart loses at most the in-flight folds, not whole experiments)."""
    metric = "pooled_ml_subject_kfold"
    subjects = [json.loads(line) for line in manifest_path.open() if line.strip()]
    by_exp: dict[str, list] = defaultdict(list)
    for s in subjects:
        if s.get("trials"):
            by_exp[s["experiment"]].append(s)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()  # (experiment, fold) pairs already written
    if out_path.exists():
        for line in out_path.open():
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    done.add((d["experiment"], d["fold"]))
                except Exception:
                    pass
    payloads = []
    for exp in sorted(by_exp):
        subs = by_exp[exp]
        k_eff = min(kfolds, len(subs))
        for f_i in range(k_eff):
            if (exp, f_i) not in done:
                payloads.append((model, exp, subs, kfolds, seed, n_starts, maxiter, min_train_subjects, f_i))
    if done:
        print(f"  [resume] {len(done)} (exp,fold) ya hechos; faltan {len(payloads)} tareas")

    n_ok = n_excl = 0

    def _emit(outp, exp, f_i, rows, nok, nex):
        nonlocal n_ok, n_excl
        for r in rows:
            outp.write(json.dumps(r) + "\n")
        outp.flush()
        n_ok += nok; n_excl += nex
        print(f"  [{exp} f{f_i}] rows={len(rows)} ok+={nok} excl+={nex}", flush=True)

    with out_path.open("a" if done else "w") as outp:
        if jobs and jobs > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = [ex.submit(_process_fold, p) for p in payloads]
                for fut in as_completed(futs):
                    exp, f_i, rows, nok, nex = fut.result()
                    _emit(outp, exp, f_i, rows, nok, nex)
        else:
            for p in payloads:
                exp, f_i, rows, nok, nex = _process_fold(p)
                _emit(outp, exp, f_i, rows, nok, nex)
    extra = f" ({n_excl} excluded/failed)" if n_excl else ""
    print(f"✓ pooled-ML K-fold: {n_ok} held-out subject NLLs (this run) → {out_path}{extra}  [metric={metric} jobs={jobs}]")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="IGT cognitive baselines (VSE/ORL/PVL-Δ) per-subject MLE. "
                    "Default = in-sample full-session (robustness). With "
                    "--fit-trials/--fit-frac = out-of-sample subject-adaptive temporal "
                    "holdout (H1 predictive metric, comparable to Centaur on the same suffix).")
    ap.add_argument("model", choices=["vse", "orl", "pvldelta"], help="model to fit")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-starts", type=int, default=8)
    # Temporal-holdout (H1 out-of-sample). The window policy IS the H1 estimand. Pick ONE.
    ap.add_argument("--fit-trials", type=int, default=None,
                    help="fit on first K trials, evaluate held-out NLL on the rest (absolute window).")
    ap.add_argument("--fit-frac", type=float, default=None,
                    help="fit on first floor(FRAC*n_trials) trials (relative; consistent across cohorts of different length).")
    ap.add_argument("--min-eval-trials", type=int, default=1,
                    help="exclude subjects with fewer than this many held-out trials.")
    ap.add_argument("--min-fit-trials", type=int, default=5,
                    help="exclude subjects whose fit window is smaller than this.")
    ap.add_argument("--maxiter", type=int, default=200,
                    help="L-BFGS-B maxiter per start (final H1 run: >=1000).")
    # H1 PRIMARY metric: pooled-ML held-out (faithful Binz). Mutually exclusive with the
    # per-subject temporal-holdout flags above.
    ap.add_argument("--pooled-kfold-subjects", action="store_true",
                    help="pooled-ML held-out participants via within-experiment subject K-fold (H1 primary).")
    ap.add_argument("--kfolds", type=int, default=5, help="K for the within-experiment subject K-fold.")
    ap.add_argument("--seed", type=int, default=20260506, help="deterministic fold/optimizer seed.")
    ap.add_argument("--min-train-subjects", type=int, default=2,
                    help="exclude a fold whose pooled training set has fewer subjects than this.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="pooled-kfold: parallel processes across experiments ([revisión interna]).")
    args = ap.parse_args()
    if args.pooled_kfold_subjects:
        if args.fit_trials is not None or args.fit_frac is not None:
            ap.error("--pooled-kfold-subjects is mutually exclusive with --fit-trials/--fit-frac")
        fit_dataset_pooled_kfold(
            args.model, args.manifest, args.out, kfolds=args.kfolds, seed=args.seed,
            n_starts=args.n_starts, maxiter=args.maxiter, min_train_subjects=args.min_train_subjects,
            jobs=args.jobs,
        )
        return
    if args.fit_trials is not None and args.fit_frac is not None:
        ap.error("pass only one of --fit-trials / --fit-frac")
    if args.fit_frac is not None and not (0.0 < args.fit_frac < 1.0):
        ap.error("--fit-frac must be in (0, 1)")
    fit_dataset(
        args.model, args.manifest, args.out, n_starts=args.n_starts,
        fit_trials=args.fit_trials, fit_frac=args.fit_frac,
        min_eval_trials=args.min_eval_trials, min_fit_trials=args.min_fit_trials,
        maxiter=args.maxiter,
    )


if __name__ == "__main__":
    main()
