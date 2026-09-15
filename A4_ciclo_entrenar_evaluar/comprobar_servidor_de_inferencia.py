"""Anexo A, apartado A.4 · El ciclo completo: entrenar el adaptador y evaluar ensayo a ensayo.

Comprobación previa obligatoria antes de cualquier ejecución: envía una sesión de prueba y verifica que
el servicio responde y que devuelve las probabilidades logarítmicas de cada token de respuesta.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a real Centaur /health and /generate endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080", help="Endpoint base URL without /generate.")
    parser.add_argument("--timeout-s", type=float, default=30)
    parser.add_argument("--require-token-logprobs", action="store_true", help="Fail unless /generate returns target_token_logprobs.")
    args = parser.parse_args()

    result = probe_endpoint(args.base_url, timeout_s=args.timeout_s, require_token_logprobs=args.require_token_logprobs)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        sys.exit(1)


def probe_endpoint(base_url: str, timeout_s: float = 30, *, require_token_logprobs: bool = False) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    result: dict[str, Any] = {
            "status": "failed",
        "base_url": base_url,
        "health": None,
        "generate": None,
        "require_token_logprobs": require_token_logprobs,
        "error": "",
    }
    try:
        health = requests.get(f"{base_url}/health", timeout=timeout_s)
        result["health"] = {"status_code": health.status_code, "body": _json_or_text(health)}
        health.raise_for_status()
        readiness_error = _readiness_error(result["health"]["body"])
        if readiness_error:
            result["error"] = readiness_error
            return result

        payload = {
            "model": "marcelbinz/Llama-3.1-Centaur-70B-adapter",
            "prompt": "In this experiment, a participant must choose between <<A>> and <<B>>. The participant chooses <<",
            "target_text": "A",
            "max_new_tokens": 8,
            "temperature": 0.0,
            "return_logprobs": require_token_logprobs,
        }
        generated = requests.post(f"{base_url}/generate", json=payload, timeout=timeout_s)
        result["generate"] = {"status_code": generated.status_code, "body": _json_or_text(generated)}
        generated.raise_for_status()
        if require_token_logprobs and not _has_target_token_logprobs(result["generate"]["body"]):
            result["error"] = "Centaur /generate did not return target_token_logprobs required for Nature-compatible NLL."
            return result
    except requests.RequestException as exc:
        result["error"] = str(exc)
        return result
    result["status"] = "passed"
    return result


def _readiness_error(health_body: Any) -> str:
    if not isinstance(health_body, dict):
        return ""
    if health_body.get("ready") is False:
        upstream_health = health_body.get("upstream_health")
        if isinstance(upstream_health, dict) and upstream_health.get("ok") is False:
            detail = upstream_health.get("error") or upstream_health.get("body") or upstream_health
            return f"Centaur endpoint health is not ready; upstream health failed: {detail}"
        return "Centaur endpoint health is not ready"
    return ""


def _has_response_token_logprobs(body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    value = body.get("response_token_logprobs") or body.get("token_logprobs")
    if isinstance(value, list) and value and all(isinstance(item, int | float) for item in value):
        return True
    choices = body.get("choices")
    if isinstance(choices, list):
        return any(_has_response_token_logprobs(choice) for choice in choices if isinstance(choice, dict))
    return False


def _has_target_token_logprobs(body: Any) -> bool:
    if not isinstance(body, dict):
        return False
    value = body.get("target_token_logprobs")
    if isinstance(value, list) and value and all(isinstance(item, int | float) for item in value):
        return True
    choices = body.get("choices")
    if isinstance(choices, list):
        return any(_has_target_token_logprobs(choice) for choice in choices if isinstance(choice, dict))
    return False


def _json_or_text(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


if __name__ == "__main__":
    main()
