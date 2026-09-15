"""Anexo A, apartado A.4 · El ciclo completo: entrenar el adaptador y evaluar ensayo a ensayo.

Recorre las sesiones del corpus de evaluación, envía cada una al servicio de inferencia y guarda, para
cada ensayo, la probabilidad del token de mazo elegido sobre el vocabulario completo y las K probabilidades más
altas (K = 5 para Centaur, K = 10 para la base). Se detiene con error si alguna sesión vuelve sin sus probabilidades
por ensayo.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path
import re
import sys
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaigns.specs import CampaignSpec
from evaluation.response_scoring import score_response
from evaluation.centaur_nll import extract_response_token_logprobs, extract_target_token_logprobs
from publication import generate_publication_package
from registry import CampaignRegistry, DatasetRegistry, ModelRegistry, ReportRegistry
from reports import build_canonical_report_payload, generate_campaign_report_bundle


CAMPAIGN_ID = "real-centaur-psych101-case"
OUTPUT_DIR = Path("data_runtime/use_cases/real_centaur_psych101")
LOCAL_ENV_PATH = Path("infra/env/centaur.env.local")
REQUESTS_PATH = OUTPUT_DIR / "centaur_requests.json"
RESPONSES_PATH = OUTPUT_DIR / "centaur_responses.json"
TRIALS_PATH = OUTPUT_DIR / "real_centaur_trials.json"
METRICS_PATH = OUTPUT_DIR / "real_centaur_metrics.json"
METRICS_MD_PATH = OUTPUT_DIR / "real_centaur_metrics.md"
BENCHMARK_REPORT_PATH = OUTPUT_DIR / "real_centaur_benchmark_report.md"
SUMMARY_PATH = OUTPUT_DIR / "real_centaur_case_summary.json"
SUMMARY_MD_PATH = OUTPUT_DIR / "real_centaur_case_summary.md"


class CentaurConfigurationError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> None:
    _load_local_env()
    args = _parse_args(argv)
    run_id = args.run_id or _default_run_id(args)
    run_output_dir = OUTPUT_DIR / "runs" / run_id
    endpoint = os.getenv("CENTAUR_ENDPOINT", "").strip()
    if args.check_config:
        _print_config_status(endpoint, args.model_name)
        return
    if not endpoint:
        raise CentaurConfigurationError(
            "CENTAUR_ENDPOINT is required for the real Centaur case. "
            "This runner does not fall back to synthetic simulation."
        )

    prompt_rows = _load_prompts(args.input_path, limit=args.limit)
    if not prompt_rows:
        raise ValueError(f"No prompts with <<...>> choices found in {args.input_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared_prompts = [_prepare_prediction_text(row, max_observed_choices=args.max_observed_choices) for row in prompt_rows]
    requests_payload = [
        _build_request_payload(
            item["prediction_prompt"],
            args.model_name,
            target_text=str(item["expected_choice"]),
            per_trial_prompt=item["evaluation_prompt"] if args.per_trial_logprobs else None,
        )
        for item in prepared_prompts
    ]
    responses = _call_centaur_with_checkpoint(
        endpoint,
        requests_payload,
        timeout_s=args.timeout_s,
        checkpoint_path=run_output_dir / "centaur_responses.partial.json",
    )
    trials = [
        _build_trial(index, prepared_prompt, response, args.model_name)
        for index, (prepared_prompt, response) in enumerate(zip(prepared_prompts, responses))
    ]
    if args.require_token_logprobs:
        missing = [trial["trial_index"] for trial in trials if not trial["metadata"].get("target_token_logprobs")]
        if missing:
            raise CentaurConfigurationError(
                "Nature-compatible runs require target-choice token logprobs for every trial. "
                f"Missing logprobs for trial indices: {missing[:10]}"
            )
    if args.per_trial_logprobs:
        missing_per_trial = [trial["trial_index"] for trial in trials if not trial["metadata"].get("per_trial_logprobs")]
        if missing_per_trial:
            raise CentaurConfigurationError(
                "Per-trial runs require per_trial_logprobs for every trial row. "
                f"Missing per-trial logprobs for trial indices: {missing_per_trial[:10]}"
            )

    _write_json(REQUESTS_PATH, requests_payload)
    _write_json(RESPONSES_PATH, responses)
    _write_json(TRIALS_PATH, trials)
    metrics = _build_metrics(trials)
    _write_json(METRICS_PATH, metrics)
    METRICS_MD_PATH.write_text(_render_metrics(metrics), encoding="utf-8")
    BENCHMARK_REPORT_PATH.write_text(
        _render_benchmark_report(metrics, args=args, model_name=args.model_name),
        encoding="utf-8",
    )

    campaign_row = CampaignRegistry().register(_build_campaign(endpoint, args.model_name), status="completed")
    dataset_rows = DatasetRegistry().register_many(
        [
            {
                "dataset_id": f"{CAMPAIGN_ID}-real-centaur-trials",
                "campaign_id": CAMPAIGN_ID,
                "kind": "real_centaur_trial_level",
                "path": str(TRIALS_PATH),
                "n_rows": len(trials),
                "source": str(args.input_path),
            }
        ]
    )
    model_rows = ModelRegistry().register_many(
        CAMPAIGN_ID,
        [
            {
                "model_id": args.model_name,
                "model_family": "centaur_real_endpoint",
                "description": "Real Centaur endpoint configured through CENTAUR_ENDPOINT.",
                "endpoint_configured": True,
            }
        ],
    )
    run_rows = [_build_run_row(trials, args.model_name)]
    report_payload = build_canonical_report_payload(
        campaign_row,
        run_rows,
        trials,
        dataset_rows=dataset_rows,
        reproducibility={
            "source_article": "A foundation model to predict and capture human cognition",
            "source_url": "https://www.nature.com/articles/s41586-025-09215-4",
            "dataset_reference": "https://huggingface.co/datasets/marcelbinz/Psych-101",
            "centaur_endpoint_env": "CENTAUR_ENDPOINT",
            "model_name": args.model_name,
            "input_path": str(args.input_path),
            "requests": str(REQUESTS_PATH),
            "responses": str(RESPONSES_PATH),
        },
    )
    report_bundle = generate_campaign_report_bundle(
        report_payload,
        OUTPUT_DIR / "report",
        objective="Real Centaur endpoint evaluation on Psych-101-style prompts",
    )
    report_row = ReportRegistry().register(
        {
            "report_id": f"{CAMPAIGN_ID}-report",
            "campaign_id": CAMPAIGN_ID,
            "kind": "real_centaur_psych101_report",
            "artifacts": report_bundle,
        }
    )
    publication_package = generate_publication_package(CAMPAIGN_ID, target_dir=_publication_target_dir())
    summary = {
        "campaign_id": CAMPAIGN_ID,
        "status": "completed",
        "scope": "real_centaur_endpoint_case",
        "synthetic_fallback_used": False,
        "model_name": args.model_name,
        "run_id": run_id,
        "counts": {"prompts": len(prompt_rows), "trials": len(trials)},
        "metrics": metrics["global"],
        "outputs": {
            "requests": str(REQUESTS_PATH),
            "responses": str(RESPONSES_PATH),
            "trials": str(TRIALS_PATH),
            "metrics": str(METRICS_PATH),
            "metrics_markdown": str(METRICS_MD_PATH),
            "benchmark_report": str(BENCHMARK_REPORT_PATH),
            "report_bundle": report_bundle,
            "publication_package": publication_package,
            "summary": str(SUMMARY_PATH),
            "summary_markdown": str(SUMMARY_MD_PATH),
        },
        "registered": {
            "campaign": campaign_row,
            "datasets": dataset_rows,
            "models": model_rows,
            "report": report_row,
        },
    }
    _write_json(SUMMARY_PATH, summary)
    SUMMARY_MD_PATH.write_text(_render_summary(summary), encoding="utf-8")
    _archive_run_outputs(run_output_dir)
    _update_run_index(run_id, run_output_dir, summary)
    print(str(SUMMARY_PATH))


def _load_local_env(path: Path | None = None) -> None:
    if path is None:
        path = LOCAL_ENV_PATH
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _publication_target_dir() -> Path:
    return OUTPUT_DIR / "publication_registry"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real Centaur Psych-101 case through an external endpoint.")
    parser.add_argument("--input-path", type=Path, default=Path("datasets/psych101/sample_psych101.jsonl"))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of prompts to load from --input-path. 0 (default) means no limit (load all rows).",
    )
    parser.add_argument("--model-name", default=os.getenv("CENTAUR_MODEL_NAME", "marcelbinz/Llama-3.1-Centaur-70B"))
    parser.add_argument("--run-id", default="", help="Optional stable run id for archiving outputs under data_runtime/use_cases/real_centaur_psych101/runs/.")
    parser.add_argument("--timeout-s", type=float, default=float(os.getenv("CENTAUR_TIMEOUT_S", "120")))
    parser.add_argument(
        "--max-observed-choices",
        type=int,
        default=64,
        help="Maximum observed <<...>> choices to keep before masking the final choice.",
    )
    parser.add_argument("--check-config", action="store_true", help="Print Centaur configuration status without calling the endpoint.")
    parser.add_argument(
        "--require-token-logprobs",
        action="store_true",
        help="Fail the run unless every response includes response-token logprobs required for Nature-compatible NLL.",
    )
    parser.add_argument(
        "--per-trial-logprobs",
        action="store_true",
        help="Ask the endpoint to score every <<...>> choice in the evaluation prompt and archive per_trial_logprobs.",
    )
    return parser.parse_args(argv)


def _print_config_status(endpoint: str, model_name: str) -> None:
    token_present = bool(os.getenv("CENTAUR_TOKEN") or os.getenv("HF_TOKEN"))
    payload = {
        "centaur_endpoint_configured": bool(endpoint),
        "centaur_endpoint": endpoint,
        "auth_token_present": token_present,
        "model_name": model_name,
        "ready_for_real_run": bool(endpoint),
        "synthetic_fallback_allowed": False,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _default_run_id(args: argparse.Namespace) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", args.input_path.stem).strip("-").lower()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stem}-n{args.limit}-w{args.max_observed_choices}-{timestamp}"


def _archive_run_outputs(run_output_dir: Path) -> None:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    for path in [
        REQUESTS_PATH,
        RESPONSES_PATH,
        TRIALS_PATH,
        METRICS_PATH,
        METRICS_MD_PATH,
        BENCHMARK_REPORT_PATH,
        SUMMARY_PATH,
        SUMMARY_MD_PATH,
    ]:
        if path.exists():
            (run_output_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _call_centaur_with_checkpoint(
    endpoint: str,
    requests_payload: list[dict[str, Any]],
    *,
    timeout_s: float,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    responses = _load_response_checkpoint(checkpoint_path)
    if len(responses) > len(requests_payload):
        # Hard abort: the checkpoint has more rows than the current input would
        # produce. Silently truncating it has destroyed multi-hour runs in the
        # past (forgetting --limit on a 720-row job reset a 691-row checkpoint
        # to 2). Force the operator to either pass the right --limit or move
        # the checkpoint aside on purpose.
        raise RuntimeError(
            f"Checkpoint at {checkpoint_path} has {len(responses)} responses "
            f"but the current input only yields {len(requests_payload)} prompts. "
            "Refusing to overwrite. Re-run with the correct --limit / --input-path, "
            "or move the checkpoint aside if you really want to start from scratch."
        )
    for payload in requests_payload[len(responses) :]:
        responses.append(_call_centaur(endpoint, payload, timeout_s=timeout_s))
        _write_json(checkpoint_path, responses)
    return responses


def _load_response_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _update_run_index(run_id: str, run_output_dir: Path, summary: dict[str, Any]) -> None:
    index_path = OUTPUT_DIR / "runs" / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"runs": []}
    row = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "path": str(run_output_dir),
        "model_name": summary["model_name"],
        "counts": summary["counts"],
        "metrics": summary["metrics"],
        "outputs": {
            "summary": str(run_output_dir / SUMMARY_PATH.name),
            "metrics_markdown": str(run_output_dir / METRICS_MD_PATH.name),
            "benchmark_report": str(run_output_dir / BENCHMARK_REPORT_PATH.name),
        },
    }
    index["runs"] = [existing for existing in index.get("runs", []) if existing.get("run_id") != run_id]
    index["runs"].append(row)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_prompts(input_path: Path, *, limit: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for raw in input_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        text = str(row.get("text") or row.get("prompt") or row.get("source_text") or "")
        if "<<" not in text or ">>" not in text:
            continue
        prompts.append(
            {
                "text": text,
                "experiment": row.get("experiment"),
                "participant": row.get("participant"),
                "dataset_name": row.get("dataset_name"),
                "split": row.get("split"),
            }
        )
        if limit > 0 and len(prompts) >= limit:
            break
    return prompts


def _build_request_payload(
    prompt: str,
    model_name: str,
    *,
    target_text: str | None = None,
    per_trial_prompt: str | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "target_text": target_text,
        "max_new_tokens": 8,
        "temperature": 0.0,
        "return_logprobs": True,
    }
    if per_trial_prompt:
        payload["return_per_trial_logprobs"] = True
        payload["per_trial_prompt"] = per_trial_prompt
    return payload


def _prepare_prediction_text(row: dict[str, Any] | str, *, max_observed_choices: int) -> dict[str, Any]:
    if isinstance(row, str):
        row = {"text": row}
    text = str(row.get("text", ""))
    compacted = _compact_to_tail_choices(text, max_observed_choices=max_observed_choices)
    return {
        "source_prompt": text,
        "evaluation_prompt": compacted,
        "prediction_prompt": _prediction_prompt(compacted),
        "expected_choice": _last_delimited_choice(compacted),
        "experiment": row.get("experiment"),
        "participant": row.get("participant"),
        "dataset_name": row.get("dataset_name"),
        "split": row.get("split"),
        "source_choice_count": text.count("<<"),
        "evaluation_choice_count": compacted.count("<<"),
        "compacted": compacted != text,
    }


def _compact_to_tail_choices(text: str, *, max_observed_choices: int) -> str:
    if max_observed_choices <= 0 or text.count("<<") <= max_observed_choices:
        return text
    lines = text.splitlines()
    first_choice_index = next((index for index, line in enumerate(lines) if "<<" in line and ">>" in line), 0)
    header = lines[:first_choice_index]
    choice_lines = [line for line in lines[first_choice_index:] if "<<" in line and ">>" in line]
    tail = choice_lines[-max_observed_choices:]
    return "\n".join([*header, "[... earlier Psych-101 trials omitted for bounded real evaluation ...]", *tail])


def _prediction_prompt(text: str) -> str:
    matches = list(re.finditer(r"<<\s*(.*?)\s*>>", text))
    if not matches:
        return text
    final = matches[-1]
    return text[: final.start()] + "<<"


def _call_centaur(endpoint: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    """POST a single prompt to the Centaur/Llama endpoint with transient-error retries.

    Retries on HTTPError 5xx, ConnectionError, Timeout — the typical failures from
    cloudflared tunnel flapping or FastAPI brief unresponsiveness on Lightning.
    Backoff: 5s, 15s, 45s, 120s (4 retries total). After all retries, propagates.

    A 404 against the HuggingFace public model page is NOT retried (configuration error,
    not transient).
    """
    headers = {"Content-Type": "application/json"}
    token = os.getenv("CENTAUR_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    backoffs = [5, 15, 45, 120]  # seconds between retries (4 retries total)
    last_exc: Exception | None = None
    for attempt in range(len(backoffs) + 1):
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_s)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                if response.status_code == 404 and "api-inference.huggingface.co/models/marcelbinz/Llama-3.1-Centaur-70B" in endpoint:
                    raise CentaurConfigurationError(
                        "The public Hugging Face model page is not a usable Centaur endpoint. "
                        "marcelbinz/Llama-3.1-Centaur-70B is not deployed by a public Inference Provider; "
                        "configure CENTAUR_ENDPOINT with your own hosted endpoint or local GPU service."
                    ) from exc
                # 5xx codes are typically transient (tunnel flap, FastAPI brief issue) -> retry.
                # 4xx codes (other than 404 special-case) are usually request errors -> do not retry.
                if 500 <= response.status_code < 600:
                    raise  # caught below by transient handler
                raise  # 4xx -> propagate immediately, no retry
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Centaur endpoint must return a JSON object")
            return data
        except CentaurConfigurationError:
            raise  # never retry
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt >= len(backoffs):
                # Last attempt; propagate.
                raise
            wait_s = backoffs[attempt]
            print(
                f"[retry] {type(exc).__name__} on attempt {attempt + 1}/{len(backoffs) + 1}; "
                f"sleeping {wait_s}s then retrying. Detail: {exc}",
                flush=True,
            )
            time.sleep(wait_s)
    # Unreachable; the loop either returns or raises.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("retry loop exited unexpectedly")


def _build_trial(index: int, prepared_prompt: dict[str, Any], response: dict[str, Any], model_name: str) -> dict[str, Any]:
    expected = str(prepared_prompt["expected_choice"])
    predicted = _extract_choice(response)
    score = score_response(expected, predicted)
    logprob = _extract_logprob(response)
    token_logprobs = extract_response_token_logprobs(response)
    target_token_logprobs = extract_target_token_logprobs(response)
    per_trial_logprobs = response.get("per_trial_logprobs")
    if not isinstance(per_trial_logprobs, list):
        per_trial_logprobs = []
    return {
        "run_id": f"{CAMPAIGN_ID}-real-centaur",
        "trial_index": index,
        "subject_id": model_name,
        "participant_type": "ai",
        "mode": "real_centaur_endpoint",
        "task_name": "psych101_prompt_prediction",
        "environment_name": "psych101_real_endpoint",
        "condition_id": "real_centaur_psych101",
        "stimulus": {
            "source_text": prepared_prompt["source_prompt"],
            "evaluation_prompt": prepared_prompt["evaluation_prompt"],
            "prediction_prompt": prepared_prompt["prediction_prompt"],
        },
        "action": predicted,
        "response_text": predicted,
        "reward": 1.0 if predicted == expected else 0.0,
        "correct": predicted == expected if expected else None,
        "latency_ms": None,
        "metadata": {
            "expected_choice": expected,
            "response_score": score,
            "experiment": prepared_prompt.get("experiment"),
            "participant": prepared_prompt.get("participant"),
            "dataset_name": prepared_prompt.get("dataset_name"),
            "split": prepared_prompt.get("split"),
            "source_choice_count": prepared_prompt["source_choice_count"],
            "evaluation_choice_count": prepared_prompt["evaluation_choice_count"],
            "prompt_compacted": prepared_prompt["compacted"],
            "prediction_prompt_masks_final_choice": prepared_prompt["prediction_prompt"] != prepared_prompt["evaluation_prompt"],
            "model_name": model_name,
            "real_centaur_endpoint": True,
            "synthetic": False,
            "logprob": logprob,
            "response_token_logprobs": token_logprobs,
            "target_text": expected,
            "target_token_logprobs": target_token_logprobs,
            "per_trial_logprobs": per_trial_logprobs,
            "per_trial_nll_ready": bool(per_trial_logprobs),
            "nature_nll_ready": bool(target_token_logprobs),
            "raw_response": response,
        },
    }


def _last_delimited_choice(text: str) -> str:
    matches = re.findall(r"<<\s*(.*?)\s*>>", text)
    return matches[-1].strip() if matches else ""


def _extract_choice(response: dict[str, Any]) -> str:
    for key in ("choice", "generated_text", "text", "response", "output"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            closing_match = re.match(r"\s*([^<>»]+?)\s*(?:>>|Â»|»)", value)
            if closing_match:
                return closing_match.group(1).strip()
            last_open = value.rfind("<<")
            if last_open >= 0:
                tail = value[last_open + 2 :]
                tail = re.split(r">>|Â»|»|\n| Trial|\.\s|,|;|:", tail.strip(), maxsplit=1)[0]
                tail = tail.strip().strip("<>").strip().strip('"').strip("'")
                if tail:
                    return tail.strip()
            match = re.search(r"<<\s*(.*?)\s*>>", value)
            if match:
                return match.group(1).strip()
            return value.strip()
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            return _extract_choice(first)
        if isinstance(first, str):
            return first.strip()
    return ""


def _extract_logprob(response: dict[str, Any]) -> float | None:
    for key in ("logprob", "log_probability"):
        value = response.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _build_campaign(endpoint: str, model_name: str) -> CampaignSpec:
    return CampaignSpec(
        campaign_id=CAMPAIGN_ID,
        title="Real Centaur Psych-101 Endpoint Case",
        description="Real Centaur endpoint execution on Psych-101-style prompts. No synthetic fallback is allowed.",
        owner="ai-system-lab",
        tags=["nature", "centaur", "psych101", "real-endpoint"],
        metadata={
            "source_article_url": "https://www.nature.com/articles/s41586-025-09215-4",
            "dataset_reference": "https://huggingface.co/datasets/marcelbinz/Psych-101",
            "endpoint": endpoint,
            "model_name": model_name,
            "synthetic_fallback_allowed": False,
        },
    )


def _build_run_row(trials: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    accuracy = [1.0 if trial.get("correct") else 0.0 for trial in trials if trial.get("correct") is not None]
    rewards = [float(trial["reward"]) for trial in trials if trial.get("reward") is not None]
    return {
        "campaign_id": CAMPAIGN_ID,
        "run_id": f"{CAMPAIGN_ID}-real-centaur",
        "mode": "real_centaur_endpoint",
        "tenant_id": "local",
        "backend": "external_endpoint",
        "condition_id": "real_centaur_psych101",
        "task_id": "psych101_prompt_prediction",
        "participant_type": "ai",
        "metrics": {
            "accuracy": sum(accuracy) / len(accuracy) if accuracy else 0.0,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "n_trials": float(len(trials)),
        },
        "metadata": {"model_name": model_name, "real_centaur_endpoint": True},
    }


def _build_metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    correct_trials = [trial for trial in trials if trial.get("correct") is not None]
    correct_count = sum(1 for trial in correct_trials if trial.get("correct") is True)
    tolerant_count = sum(
        1
        for trial in correct_trials
        if trial.get("metadata", {}).get("response_score", {}).get("tolerant_match") is True
    )
    global_metrics = {
        "n_trials": len(trials),
        "n_scored": len(correct_trials),
        "n_correct": correct_count,
        "accuracy": correct_count / len(correct_trials) if correct_trials else None,
        "n_tolerant_correct": tolerant_count,
        "tolerant_accuracy": tolerant_count / len(correct_trials) if correct_trials else None,
        "mean_reward": sum(float(trial.get("reward") or 0.0) for trial in trials) / len(trials) if trials else None,
    }

    by_experiment: dict[str, dict[str, Any]] = {}
    for trial in trials:
        experiment = str(trial.get("metadata", {}).get("experiment") or "unknown")
        bucket = by_experiment.setdefault(experiment, {"experiment": experiment, "n_trials": 0, "n_correct": 0, "n_tolerant_correct": 0})
        bucket["n_trials"] += 1
        if trial.get("correct") is True:
            bucket["n_correct"] += 1
        if trial.get("metadata", {}).get("response_score", {}).get("tolerant_match") is True:
            bucket["n_tolerant_correct"] += 1
    for bucket in by_experiment.values():
        bucket["accuracy"] = bucket["n_correct"] / bucket["n_trials"] if bucket["n_trials"] else None
        bucket["tolerant_accuracy"] = bucket["n_tolerant_correct"] / bucket["n_trials"] if bucket["n_trials"] else None

    confusion_rows = [
        {
            "trial_index": trial["trial_index"],
            "experiment": trial.get("metadata", {}).get("experiment"),
            "expected": trial.get("metadata", {}).get("expected_choice"),
            "predicted": trial.get("action"),
            "correct": trial.get("correct"),
            "tolerant_correct": trial.get("metadata", {}).get("response_score", {}).get("tolerant_match"),
            "ordinal_distance": trial.get("metadata", {}).get("response_score", {}).get("ordinal_distance"),
            "numeric_distance": trial.get("metadata", {}).get("response_score", {}).get("numeric_distance"),
            "source_choice_count": trial.get("metadata", {}).get("source_choice_count"),
            "evaluation_choice_count": trial.get("metadata", {}).get("evaluation_choice_count"),
        }
        for trial in trials
    ]
    return {
        "global": global_metrics,
        "nature_nll_ready": all(bool(trial.get("metadata", {}).get("target_token_logprobs")) for trial in trials) if trials else False,
        "n_trials_with_response_token_logprobs": sum(
            1 for trial in trials if trial.get("metadata", {}).get("response_token_logprobs")
        ),
        "n_trials_with_target_token_logprobs": sum(
            1 for trial in trials if trial.get("metadata", {}).get("target_token_logprobs")
        ),
        "by_experiment": sorted(by_experiment.values(), key=lambda row: str(row["experiment"])),
        "confusion_rows": confusion_rows,
    }


def _render_metrics(metrics: dict[str, Any]) -> str:
    lines = [
        "# Real Centaur Psych-101 Metrics",
        "",
        "## Global",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in metrics["global"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## By Experiment", "", "| experiment | n_trials | n_correct | accuracy | n_tolerant_correct | tolerant_accuracy |", "|---|---:|---:|---:|---:|---:|"])
    for row in metrics["by_experiment"]:
        lines.append(
            f"| `{row['experiment']}` | {row['n_trials']} | {row['n_correct']} | {row['accuracy']} | "
            f"{row['n_tolerant_correct']} | {row['tolerant_accuracy']} |"
        )
    lines.extend(["", "## Trial Outcomes", "", "| trial | experiment | expected | predicted | exact | tolerant |", "|---:|---|---|---|---|---|"])
    for row in metrics["confusion_rows"]:
        lines.append(
            f"| {row['trial_index']} | `{row['experiment']}` | `{row['expected']}` | `{row['predicted']}` | {row['correct']} | {row['tolerant_correct']} |"
        )
    return "\n".join(lines) + "\n"


def _render_benchmark_report(metrics: dict[str, Any], *, args: argparse.Namespace, model_name: str) -> str:
    global_metrics = metrics["global"]
    lines = [
        "# Real Centaur Psych-101 Benchmark Report",
        "",
        "## Configuration",
        "",
        f"- model: `{model_name}`",
        f"- input_path: `{args.input_path}`",
        f"- limit: `{args.limit}`",
        f"- max_observed_choices: `{args.max_observed_choices}`",
        f"- synthetic_fallback_used: `False`",
        "",
        "## Result",
        "",
        f"- n_trials: `{global_metrics['n_trials']}`",
        f"- n_correct: `{global_metrics['n_correct']}` / `{global_metrics['n_scored']}`",
        f"- accuracy: `{global_metrics['accuracy']}`",
        f"- n_tolerant_correct: `{global_metrics['n_tolerant_correct']}` / `{global_metrics['n_scored']}`",
        f"- tolerant_accuracy: `{global_metrics['tolerant_accuracy']}`",
        f"- mean_reward: `{global_metrics['mean_reward']}`",
        "",
        "## Interpretation",
        "",
        "This run is a real endpoint evaluation of Centaur on a bounded Psych-101 sample. "
        "Each row is converted into a held-out final-choice prediction: the final observed `<<...>>` answer is masked, "
        "Centaur completes it, and the completion is compared with the held-out human response.",
        "",
        "Rows with hundreds of choices are compacted by preserving the instruction header and the most recent observed choices. "
        "The full original transcript is retained in `real_centaur_trials.json` under `stimulus.source_text`; "
        "the exact prompt sent to Centaur is retained under `stimulus.prediction_prompt`.",
        "",
        "## By Experiment",
        "",
        "| experiment | n_trials | n_correct | accuracy | n_tolerant_correct | tolerant_accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["by_experiment"]:
        lines.append(
            f"| `{row['experiment']}` | {row['n_trials']} | {row['n_correct']} | {row['accuracy']} | "
            f"{row['n_tolerant_correct']} | {row['tolerant_accuracy']} |"
        )
    lines.extend(
        [
            "",
            "## Errors",
            "",
            "| trial | experiment | expected | predicted |",
            "|---:|---|---|---|",
        ]
    )
    for row in metrics["confusion_rows"]:
        if row["correct"] is False:
            lines.append(f"| {row['trial_index']} | `{row['experiment']}` | `{row['expected']}` | `{row['predicted']}` |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is a bounded smoke benchmark, not a full Psych-101 reproduction.",
            "- The sample is selected from the public training split and should not be interpreted as held-out test-set performance.",
            "- Accuracy is exact-match on the extracted final choice; graded or ordinal tasks may need task-specific scoring later.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _render_summary(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Real Centaur Psych-101 Case",
            "",
            f"- campaign_id: `{summary['campaign_id']}`",
            f"- status: `{summary['status']}`",
            f"- model_name: `{summary['model_name']}`",
            f"- prompts: `{summary['counts']['prompts']}`",
            f"- accuracy: `{summary['metrics']['accuracy']}`",
            f"- n_correct: `{summary['metrics']['n_correct']}` / `{summary['metrics']['n_scored']}`",
            f"- tolerant_accuracy: `{summary['metrics']['tolerant_accuracy']}`",
            f"- synthetic_fallback_used: `{summary['synthetic_fallback_used']}`",
            "",
            "## Outputs",
            "",
            f"- trials: `{summary['outputs']['trials']}`",
            f"- metrics: `{summary['outputs']['metrics_markdown']}`",
            f"- benchmark report: `{summary['outputs']['benchmark_report']}`",
            f"- report: `{summary['outputs']['report_bundle']['markdown']}`",
            f"- publication manifest: `{summary['outputs']['publication_package']['publication_manifest']}`",
        ]
    )


if __name__ == "__main__":
    main()
