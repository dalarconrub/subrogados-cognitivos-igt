#!/usr/bin/env python3
"""Anexo A, apartado A.4 · El ciclo completo: entrenar el adaptador y evaluar ensayo a ensayo.

Carga la base con los pesos reinicializados al azar y calcula su verosimilitud negativa por sesión sobre
el corpus, con el mismo procedimiento que los demás modelos.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    from scipy import stats as scstats
except ImportError:
    print("ERROR: numpy + scipy required.", file=sys.stderr)
    sys.exit(3)

SEED = 20260506
N_BOOTSTRAP = 10_000

# [revisión interna] §2.Q5: equivalence region aligned with H4 convention (|d| < 0.10).
EQUIVALENCE_D = 0.10
WIN_D_THRESHOLD = -0.10
ALPHA = 0.05


# ---------- utilities ----------------------------------------------------

def _safe_filename(s: str) -> str:
    """Convert HF model id (e.g. 'marcelbinz/Llama-3.1-Centaur-70B-adapter')
    into a filesystem-safe stem (e.g. 'marcelbinz__Llama-3-1-Centaur-70B-adapter').
    """
    return re.sub(r"[^A-Za-z0-9_-]", "_", s.replace("/", "__")).strip("_")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_sha() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return r.stdout.strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- HF cache cleanup ---

def _hf_cache_dir(model_id: str) -> Path:
    """Return HF cache directory path for a specific model_id."""
    safe_id = model_id.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    return cache_root / f"models--{safe_id}"


def _hf_cache_size_gb(model_id: str) -> float:
    """Return total cached size for model_id in GB."""
    cache_dir = _hf_cache_dir(model_id)
    if not cache_dir.exists():
        return 0.0
    total = 0
    for p in cache_dir.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total / 1024**3


def _clean_hf_cache(model_id: str) -> bool:
    """Delete HF cache for model_id. Returns True if deleted."""
    import shutil
    cache_dir = _hf_cache_dir(model_id)
    if not cache_dir.exists():
        print(f"  [HF CACHE] {cache_dir.name} not found (already cleaned?)")
        return False
    size_before_gb = _hf_cache_size_gb(model_id)
    try:
        shutil.rmtree(cache_dir)
        print(f"  [HF CACHE] cleaned {cache_dir.name} ({size_before_gb:.1f} GB freed)")
        return True
    except Exception as e:
        print(f"  [HF CACHE] WARN: failed to clean {cache_dir.name}: {e}")
        return False


def _disk_free_gb(path: str = "/content") -> float:
    """Return free disk GB at path (Colab /content typical)."""
    try:
        import shutil as sh
        usage = sh.disk_usage(path)
        return usage.free / 1024**3
    except Exception:
        try:
            usage = __import__("shutil").disk_usage("/")
            return usage.free / 1024**3
        except Exception:
            return -1.0


# ---------- CUDA mem diagnostics ----------------------------

def _cuda_mem_snapshot(label: str) -> dict:
    """Log + return CUDA memory state. Returns empty dict if no GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        snap = {
            "label": label,
            "free_gb": round(free / 1024**3, 2),
            "total_gb": round(total / 1024**3, 2),
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        }
        print(f"  [CUDA] {label}: free={snap['free_gb']}GB allocated={snap['allocated_gb']}GB "
              f"reserved={snap['reserved_gb']}GB peak={snap['max_allocated_gb']}GB")
        return snap
    except Exception as e:
        print(f"  [CUDA] {label}: snapshot failed: {e}")
        return {}


def _free_gpu(label_before: str = "before-cleanup", label_after: str = "after-cleanup") -> tuple[dict, dict]:
    """Aggressive sequential cleanup between 70B 4-bit model loads."""
    before = _cuda_mem_snapshot(label_before)
    try:
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        print(f"  [CUDA] cleanup warning: {e}")
    after = _cuda_mem_snapshot(label_after)
    return before, after


# ---------- IGT validation ---------------------------------

def validate_igt_data(igt: list[dict]) -> dict:
    """Validate IGT prompts: schema, marker presence, duplicate prompt_ids.

    Returns summary dict with diagnostic info. Raises on hard failures.
    """
    if not isinstance(igt, list) or not igt:
        raise ValueError("IGT data must be non-empty list")
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    missing_text = 0
    missing_marker = 0
    for r in igt:
        if not isinstance(r, dict):
            raise ValueError(f"IGT record not dict: {r!r}")
        if "prompt_id" not in r or "text" not in r:
            raise ValueError(f"IGT record missing prompt_id/text: {r!r}")
        pid = r["prompt_id"]
        if pid in seen_ids:
            duplicates.append(pid)
        seen_ids.add(pid)
        text = r.get("text", "")
        if not text:
            missing_text += 1
            continue
        if "<<" not in text or ">>" not in text:
            missing_marker += 1
    summary = {
        "n_records": len(igt),
        "n_unique_prompt_ids": len(seen_ids),
        "n_duplicates": len(duplicates),
        "n_missing_text": missing_text,
        "n_missing_markers": missing_marker,
    }
    if duplicates:
        raise ValueError(f"Duplicate prompt_ids ({len(duplicates)}): {duplicates[:5]}...")
    if missing_marker > 0:
        # IGT prompts MUST have markers for masked-loss NLL. Else NLL is undefined.
        raise ValueError(f"{missing_marker} IGT records lack <<X>> markers; eval undefined.")
    return summary


# ---------- tokenizer identity checks ----------------------

def _tokenizer_fingerprint(tok) -> dict:
    """Identity stamp for a tokenizer: vocab size + special token ids."""
    try:
        vocab_size = len(tok.get_vocab()) if hasattr(tok, "get_vocab") else getattr(tok, "vocab_size", None)
    except Exception:
        vocab_size = getattr(tok, "vocab_size", None)
    return {
        "name_or_path": getattr(tok, "name_or_path", None),
        "vocab_size": vocab_size,
        "bos_token_id": getattr(tok, "bos_token_id", None),
        "eos_token_id": getattr(tok, "eos_token_id", None),
        "pad_token_id": getattr(tok, "pad_token_id", None),
        "unk_token_id": getattr(tok, "unk_token_id", None),
    }


def _check_tokenizer_compat(tok_a: dict, tok_b: dict, label_a: str, label_b: str) -> None:
    """Hard-fail if two tokenizers differ on identity fields. [revisión interna] P1."""
    keys = ["vocab_size", "bos_token_id", "eos_token_id"]
    mismatches = {k: (tok_a[k], tok_b[k]) for k in keys if tok_a.get(k) != tok_b.get(k)}
    if mismatches:
        raise ValueError(
            f"Tokenizer mismatch {label_a} vs {label_b}: {mismatches}. "
            f"Bias risk on NLL; aborting. Pass --allow-tokenizer-fallback to override "
            f"(only do this if you have audited the difference)."
        )


# ---------- NLL computation ----------------------------------------------

def _import_collator():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                                / "stage_7_5_1_lora_noise"))
        from masked_loss_collator import compute_answer_token_nll  # noqa
        return compute_answer_token_nll
    except ImportError as e:
        print(f"ERROR: missing masked_loss_collator dep: {e}", file=sys.stderr)
        sys.exit(3)


def compute_nll_centaur(centaur_base: str, centaur_adapter: str,
                        igt_data: list[dict],
                        allow_tokenizer_fallback: bool = False) -> tuple[list[dict], dict]:
    """Load pretrained base + Centaur adapter; compute IGT answer-token NLL.

    Returns (nlls, tokenizer_fingerprint).
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        print(f"ERROR: missing dep {e}", file=sys.stderr)
        sys.exit(3)
    compute_answer_token_nll = _import_collator()

    if not torch.cuda.is_available():
        print("ERROR: requires GPU.", file=sys.stderr)
        sys.exit(3)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(centaur_base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok_fp = _tokenizer_fingerprint(tok)
    print(f"  Centaur tokenizer fp: {tok_fp}")

    _cuda_mem_snapshot("before-centaur-load")
    print(f"  Loading Centaur: {centaur_base} + {centaur_adapter}")
    m = AutoModelForCausalLM.from_pretrained(
        centaur_base, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )
    m = PeftModel.from_pretrained(m, centaur_adapter)
    m.eval()
    _cuda_mem_snapshot("after-centaur-load")

    out: list[dict] = []
    for prompt in igt_data:
        nll_info = compute_answer_token_nll(m, tok, prompt["text"])
        out.append({
            "prompt_id": prompt["prompt_id"],
            "experiment": prompt.get("experiment", "igt"),
            "cohort": prompt.get("cohort"),
            "nll": nll_info["mean_nll"],
            "n_answer_tokens": nll_info["n_answer_tokens"],
            "n_answer_spans": nll_info["n_answer_spans"],
        })

    del m, tok
    _free_gpu("before-cleanup-centaur", "after-cleanup-centaur")
    return out, tok_fp


def compute_nll_randominit(randominit_model: str, igt_data: list[dict],
                           reference_tok_fp: dict,
                           allow_tokenizer_fallback: bool = False) -> tuple[list[dict], dict]:
    """Load RandomInit-70B base alone; compute IGT answer-token NLL.

    Strict tokenizer identity check vs reference_tok_fp (Centaur's tokenizer).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        print(f"ERROR: missing dep {e}", file=sys.stderr)
        sys.exit(3)
    compute_answer_token_nll = _import_collator()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Tokenizer: try model's own; fall back to Llama-3.1-70B ONLY if --allow-tokenizer-fallback.
    used_fallback = False
    try:
        tok = AutoTokenizer.from_pretrained(randominit_model)
        tok_source = randominit_model
    except Exception as e:
        if not allow_tokenizer_fallback:
            raise RuntimeError(
                f"Tokenizer load failed for {randominit_model}: {e}. "
                f"Pass --allow-tokenizer-fallback to use meta-llama/Llama-3.1-70B tokenizer instead."
            )
        print(f"  WARN: tokenizer fallback")
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B")
        tok_source = "meta-llama/Llama-3.1-70B (fallback)"
        used_fallback = True
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    randominit_tok_fp = _tokenizer_fingerprint(tok)
    randominit_tok_fp["fallback_used"] = used_fallback
    randominit_tok_fp["tokenizer_source"] = tok_source
    print(f"  RandomInit tokenizer fp: {randominit_tok_fp}")

    # Strict identity check (raises on mismatch unless allowed)
    if not allow_tokenizer_fallback:
        _check_tokenizer_compat(reference_tok_fp, randominit_tok_fp,
                                "centaur", "randominit")
    else:
        print(f"  WARN: skipping tokenizer compat check (--allow-tokenizer-fallback)")

    _cuda_mem_snapshot("before-randominit-load")
    print(f"  Loading RandomInit-70B: {randominit_model} (NO adapter)")
    m = AutoModelForCausalLM.from_pretrained(
        randominit_model, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )
    m.eval()
    _cuda_mem_snapshot("after-randominit-load")

    out: list[dict] = []
    for prompt in igt_data:
        nll_info = compute_answer_token_nll(m, tok, prompt["text"])
        out.append({
            "prompt_id": prompt["prompt_id"],
            "experiment": prompt.get("experiment", "igt"),
            "cohort": prompt.get("cohort"),
            "nll": nll_info["mean_nll"],
            "n_answer_tokens": nll_info["n_answer_tokens"],
            "n_answer_spans": nll_info["n_answer_spans"],
        })

    del m, tok
    _free_gpu("before-cleanup-randominit", "after-cleanup-randominit")
    return out, randominit_tok_fp


# ---------- paired stats + verdict ---------------------------------------

def paired_stats(centaur_nll: list[dict], randominit_nll: list[dict]) -> dict:
    a = {x["prompt_id"]: x["nll"] for x in centaur_nll}
    b = {x["prompt_id"]: x["nll"] for x in randominit_nll}
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("No paired prompts (no overlap)")
    deltas = np.array([a[p] - b[p] for p in common], dtype=float)
    t_stat, p_two = scstats.ttest_1samp(deltas, 0.0)
    p_one = (p_two / 2) if t_stat < 0 else 1.0 - (p_two / 2)
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1))
    d = mean / sd if sd > 0 else float("nan")
    rng = np.random.default_rng(SEED)
    boot = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].mean()
                     for _ in range(N_BOOTSTRAP)])
    return {
        "n": len(deltas),
        "delta_mean_centaur_minus_randominit": mean,
        "delta_sd": sd,
        "ci_lo_95": float(np.quantile(boot, 0.025)),
        "ci_hi_95": float(np.quantile(boot, 0.975)),
        "t_stat": float(t_stat),
        "df": len(deltas) - 1,
        "p_value_one_sided": float(p_one),
        "cohens_d": d,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "equivalence_d_threshold": EQUIVALENCE_D,
        "win_d_threshold": WIN_D_THRESHOLD,
        "alpha": ALPHA,
    }


def verdict_c25(stats: dict, training_status: str) -> str:
    """[revisión interna] verdict. [revisión interna] §2.Q1/Q2: label discipline by training_status.

    - If untrained_random_base: rename verdict to "lower_bound_check" to avoid
      overclaiming "pretraining matters": the contrast with the untrained reference is descriptive.
    - If randominit_trained_control: use "pretraining_matters" framing.
    - If unknown: conservative "lower_bound_check".
    """
    win = (stats["delta_mean_centaur_minus_randominit"] < 0
           and stats["p_value_one_sided"] < ALPHA
           and stats["cohens_d"] < WIN_D_THRESHOLD)
    equiv = abs(stats["cohens_d"]) < EQUIVALENCE_D

    if win:
        if training_status == "randominit_trained_control":
            return "centaur_beats_randominit__pretraining_matters"
        else:
            return "centaur_beats_randominit__lower_bound_check"
    if equiv:
        if training_status == "randominit_trained_control":
            return "centaur_equivalent_to_randominit__pretraining_irrelevant"
        else:
            return "centaur_equivalent_to_randominit__unexpected_check_pipeline"
    return "inconclusive"


# ---------- cache manifest ---------------------------------

def _cache_manifest_validate(manifest_path: Path, expected: dict) -> None:
    """Raise if cache manifest doesn't match expected args."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest missing: {manifest_path}")
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    drift = {k: (saved.get(k), v) for k, v in expected.items()
             if saved.get(k) != v}
    if drift:
        raise ValueError(
            f"Cache manifest mismatch (saved vs requested): {drift}. "
            f"Delete cache to regenerate OR fix arguments."
        )


def _cache_manifest_write(manifest_path: Path, content: dict) -> None:
    manifest_path.write_text(json.dumps(content, indent=2, ensure_ascii=False),
                             encoding="utf-8")


# ---------- main ---------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--centaur-base", default="meta-llama/Llama-3.1-70B")
    p.add_argument("--centaur-adapter", default="marcelbinz/Llama-3.1-Centaur-70B-adapter")
    p.add_argument("--randominit-model", default="marcelbinz/Llama-3.1-RandomInit-70B")
    p.add_argument("--randominit-training-status",
                   default="untrained_random_base",
                   choices=["untrained_random_base", "randominit_trained_control", "unknown"],
                   help="[revisión interna] P0: identity of the random-init artifact. "
                        "Default 'untrained_random_base' per HF account inspection "
                        "(no RandomInit-70B-Centaur-adapter sibling exists).")
    p.add_argument("--igt-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument
    p.add_argument("--allow-tokenizer-fallback", action="store_true",
                   help="[revisión interna] P1: explicit opt-in for tokenizer fallback "
                        "(default Llama-3.1-70B) if model's own tokenizer missing.")
    p.add_argument("--clean-hf-cache-between-phases", action="store_true",
                   default=True,
                   help="[revisión interna] disk-constraint patch: delete cached "
                        "Llama-3.1-70B base after [1/3] Centaur done; "
                        "RandomInit-70B is self-contained 70B model so doesn't "
                        "need Llama-base on disk. Frees ~140 GB. Default True "
                        "for Colab compatibility (235 GB /content limit).")
    p.add_argument("--no-clean-hf-cache", dest="clean_hf_cache_between_phases",
                   action="store_false",
                   help="Skip HF cache cleanup. Use only if disk budget >280 GB.")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if not args.igt_data.exists():
        print(f"ERROR: IGT data not found: {args.igt_data}", file=sys.stderr)
        return 1

    # IGT validation (P2)
    igt_prompts = json.loads(args.igt_data.read_text(encoding="utf-8"))
    igt_summary = validate_igt_data(igt_prompts)
    igt_sha = _file_sha256(args.igt_data)
    print(f"IGT validation: {igt_summary}")
    print(f"IGT SHA256: {igt_sha}")

    # Cache stamping (P1)
    cache_centaur_path = args.output / f"nll_centaur__{_safe_filename(args.centaur_base)}__{_safe_filename(args.centaur_adapter)}.json"
    cache_randominit_path = args.output / f"nll_randominit__{_safe_filename(args.randominit_model)}.json"
    cache_manifest_path = args.output / "cache_manifest.json"

    expected_manifest = {
        "centaur_base": args.centaur_base,
        "centaur_adapter": args.centaur_adapter,
        "randominit_model": args.randominit_model,
        "igt_data_sha256": igt_sha,
    }

    # Compute or load NLLs
    ran_centaur = False
    ran_randominit = False
    centaur_tok_fp = randominit_tok_fp = None

    if args.reuse_cached_nll:
        _cache_manifest_validate(cache_manifest_path, expected_manifest)
        print("[1/3] Reusing cached Centaur NLL (manifest validated)")
        centaur_nll = json.loads(cache_centaur_path.read_text(encoding="utf-8"))
        print("[2/3] Reusing cached RandomInit NLL (manifest validated)")
        randominit_nll = json.loads(cache_randominit_path.read_text(encoding="utf-8"))
        # Tokenizer fingerprints come from previous manifest
        saved = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        centaur_tok_fp = saved.get("centaur_tokenizer_fingerprint", {})
        randominit_tok_fp = saved.get("randominit_tokenizer_fingerprint", {})
    else:
        print("[1/3] NLL Centaur (Llama-3.1-70B base + Centaur adapter)...")
        centaur_nll, centaur_tok_fp = compute_nll_centaur(
            args.centaur_base, args.centaur_adapter, igt_prompts,
            allow_tokenizer_fallback=args.allow_tokenizer_fallback,
        )
        cache_centaur_path.write_text(json.dumps(centaur_nll, indent=2), encoding="utf-8")
        ran_centaur = True

        # [revisión interna] post-test patch: clean HF cache of Llama-base after Centaur NLL
        # done. RandomInit-70B is self-contained 70B model (no LoRA adapter); doesn't
        # need Llama-base on disk. Frees ~140 GB to make room for RandomInit download
        # within Colab 235 GB /content limit.
        if args.clean_hf_cache_between_phases:
            print(f"  [HF CACHE] Pre-cleanup disk free: {_disk_free_gb():.1f} GB")
            _clean_hf_cache(args.centaur_base)
            print(f"  [HF CACHE] Post-cleanup disk free: {_disk_free_gb():.1f} GB")

        print("[2/3] NLL RandomInit-70B (no adapter)...")
        randominit_nll, randominit_tok_fp = compute_nll_randominit(
            args.randominit_model, igt_prompts,
            reference_tok_fp=centaur_tok_fp,
            allow_tokenizer_fallback=args.allow_tokenizer_fallback,
        )
        cache_randominit_path.write_text(json.dumps(randominit_nll, indent=2), encoding="utf-8")
        ran_randominit = True

    # [3/3] Paired stats + verdict
    print
    stats = paired_stats(centaur_nll, randominit_nll)
    stats["verdict_c25"] = verdict_c25(stats, args.randominit_training_status)
    stats["randominit_training_status"] = args.randominit_training_status

    # Provenance (P2)
    cache_centaur_sha = _file_sha256(cache_centaur_path)
    cache_randominit_sha = _file_sha256(cache_randominit_path)
    provenance = {
        "tier": "sensibilidad",
        "reporting_location": "[revisión interna] [revisión interna] / anexo",
        "script_path": str(Path(__file__).relative_to(Path(__file__).resolve().parents[5])
                           if Path(__file__).is_absolute() else Path(__file__)),
        "script_commit": _git_head_sha(),
        "generation_date_utc": _now_iso(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "peft": _pkg_version("peft"),
            "bitsandbytes": _pkg_version("bitsandbytes"),
            "datasets": _pkg_version("datasets"),
        },
        "centaur": {
            "base_model_id": args.centaur_base,
            "adapter_id": args.centaur_adapter,
            "tokenizer_fingerprint": centaur_tok_fp,
        },
        "randominit": {
            "model_id": args.randominit_model,
            "training_status": args.randominit_training_status,
            "tokenizer_fingerprint": randominit_tok_fp,
        },
        "igt_data": {
            "path": str(args.igt_data),
            "sha256": igt_sha,
            "validation_summary": igt_summary,
        },
        "cache_files": {
            "centaur_nll": {
                "path": str(cache_centaur_path),
                "sha256": cache_centaur_sha,
                "computed_this_run": ran_centaur,
            },
            "randominit_nll": {
                "path": str(cache_randominit_path),
                "sha256": cache_randominit_sha,
                "computed_this_run": ran_randominit,
            },
        },
        "thresholds": {
            "win_d": WIN_D_THRESHOLD,
            "equivalence_d": EQUIVALENCE_D,
            "alpha": ALPHA,
        },
    }
    stats["provenance"] = provenance

    out_path = args.output / "eval_centaur_vs_randominit_stats.json"
    out_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # Cache manifest (P1)
    _cache_manifest_write(cache_manifest_path, {
        **expected_manifest,
        "centaur_tokenizer_fingerprint": centaur_tok_fp,
        "randominit_tokenizer_fingerprint": randominit_tok_fp,
        "generation_date_utc": _now_iso(),
        "script_commit": _git_head_sha(),
    })

    print(f"\n=== [revisión interna] sensitivity summary ([revisión interna] anexo) ===")
    print(f"  Centaur NLL vs RandomInit-70B NLL — paired Δ = {stats['delta_mean_centaur_minus_randominit']:+.4f}")
    print(f"  Cohen's d (signed)        = {stats['cohens_d']:+.3f}")
    print(f"  CI 95% bootstrap          = [{stats['ci_lo_95']:+.4f}, {stats['ci_hi_95']:+.4f}]")
    print(f"  p-value one-sided         = {stats['p_value_one_sided']:.2e}")
    print(f"  RandomInit training_status = {args.randominit_training_status}")
    print
    print(f"\nOK: outputs at {args.output}")
    print(f"    stats: {out_path.name}")
    print(f"    cache manifest: {cache_manifest_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
