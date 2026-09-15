"""Anexo A, apartado A.3 · Formato de los corpus.

Convierte cada sesión de los estudios de origen al formato de texto de Psych-101: la instrucción de la
tarea seguida del historial ensayo a ensayo, con la elección entre los marcadores “<<” y “>>”. Asigna a cada
sesión cuatro letras de mazo al azar, con una semilla derivada del identificador del sujeto, y conserva la
correspondencia con la codificación A-D de la literatura.
"""


from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path("<REPO>")
IN_DIR = REPO_ROOT / "data_runtime" / "plans" / "dm_paper1"
OUT_DIR = IN_DIR  # same directory

# Letters to use as random deck labels (alphabet minus ambiguous I/O/S)
DECK_LABEL_POOL = list("ABCDEFGHJKLMNPQRTUVWXYZ")
ORIG_DECKS = ["A", "B", "C", "D"]

# Per-dataset loan amount + canonical trial count (matches IGT version)
DATASET_PARAMS = {
    "ahn2014": {"loan": 2000, "n_trials_canonical": 100, "currency": "$"},
    "kildahl2020": {"loan": 2000, "n_trials_canonical": 100, "currency": "$"},
    "steingroever2015": {"loan": 2000, "n_trials_canonical": None, "currency": "$"},  # 95/100/150 variant
    "sullivantoole2022": {"loan": 2000, "n_trials_canonical": 100, "currency": "$"},
    "chavez2026": {"loan": 2000, "n_trials_canonical": 200, "currency": "$"},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def deck_label_map(subject_id: str) -> dict[str, str]:
    """Deterministic per-subject randomization of A/B/C/D → random letters.

    Uses SHA256 of subject_id as seed so the same subject always gets the same
    mapping, and different subjects get independent randomizations.
    """
    seed = int(hashlib.sha256(subject_id.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    chosen = rng.sample(DECK_LABEL_POOL, 4)
    return dict(zip(ORIG_DECKS, chosen))


def format_prompt(subject: dict, params: dict) -> str:
    """Build the Psych-101 IGT prompt text for one subject."""
    n_trials = subject["n_trials"]
    loan = params["loan"]
    currency = params["currency"]
    mapping = deck_label_map(subject["subject_id"])
    inv = {v: k for k, v in mapping.items()}  # not used but useful for inspection
    labels = [mapping[d] for d in ORIG_DECKS]

    preamble = (
        f"You see in front of you four decks of cards labeled "
        f"{labels[0]}, {labels[1]}, {labels[2]}, and {labels[3]}.\n"
        f"You get a loan of {loan}{currency} of play money.\n"
        f"You have to select one card at a time, from any of the four decks, "
        f"for {n_trials} trials.\n"
        "You select a card from a deck by pressing the corresponding key.\n"
        "After turning a card, you win some money, the amount varies with the deck.\n"
        "You sometimes also have to pay a penalty, which also varies with the deck.\n"
        "Your goal is to maximize profit on the loan of the play money.\n"
        "\n"
    )

    trial_lines = []
    for t in subject["trials"]:
        deck = mapping.get(t["deck"], "?")
        gain = float(t.get("gain", 0.0))
        loss = abs(float(t.get("loss", 0.0)))  # display as positive magnitude
        trial_lines.append(
            f"You press <<{deck}>>. You win {gain:.1f}{currency} and lose {loss:.1f}{currency}."
        )
    return preamble + "\n".join(trial_lines)


def build_manifest_for_dataset(dataset: str) -> tuple[Path, int, int, str]:
    src = IN_DIR / f"igt_{dataset}_subject_data.jsonl"
    if not src.exists():
        print(f"[{dataset}] missing analysis manifest: {src}")
        return src, 0, 0, ""
    params = DATASET_PARAMS[dataset]
    out_path = IN_DIR / f"igt_{dataset}_inference_manifest.jsonl"

    n_subjects = 0
    n_trials_total = 0
    with src.open() as inp, out_path.open("w") as outp:
        for line in inp:
            subj = json.loads(line)
            text = format_prompt(subj, params)
            # `experiment` is per-cohort (used by runner for cluster bootstrap grouping).
            # `participant` is the subject identifier (runner reads this name).
            row = {
                "manifest": subj["manifest"],
                "experiment": f"igt_{subj['manifest']}",
                "participant": subj["subject_id"],
                "subject_id": subj["subject_id"],  # kept for backwards-compat with our analysis scripts
                "cohort": subj.get("cohort"),
                "session": subj.get("session"),
                "igt_version": subj["igt_version"],
                "n_trials": subj["n_trials"],
                "text_len_chars": len(text),
                "text": text,
                "deck_label_map": deck_label_map(subj["subject_id"]),
            }
            outp.write(json.dumps(row) + "\n")
            n_subjects += 1
            n_trials_total += subj["n_trials"]
    sha = sha256_file(out_path)
    print(f"[{dataset}] {n_subjects} subjects, {n_trials_total} trials → {out_path}")
    print(f"           sha256: {sha[:16]}...")
    return out_path, n_subjects, n_trials_total, sha


def main() -> None:
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": "Psych-101 IGT prompt template with per-subject deterministic deck-label randomization",
        "datasets": {},
        "total_subjects": 0,
        "total_trials": 0,
    }
    for ds in DATASET_PARAMS:
        path, n_subj, n_trials, sha = build_manifest_for_dataset(ds)
        if n_subj == 0:
            continue
        summary["datasets"][ds] = {
            "output_path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha,
            "n_subjects": n_subj,
            "n_trials": n_trials,
            "avg_chars_per_prompt": None,
        }
        # Read back avg chars for summary
        with path.open() as f:
            chars = [json.loads(l)["text_len_chars"] for l in f]
        if chars:
            summary["datasets"][ds]["avg_chars_per_prompt"] = sum(chars) / len(chars)
            summary["datasets"][ds]["max_chars"] = max(chars)
            summary["datasets"][ds]["min_chars"] = min(chars)
        summary["total_subjects"] += n_subj
        summary["total_trials"] += n_trials

    summary_path = OUT_DIR / "igt_combined_inference_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Combined inference manifest summary: {summary_path}")
    print(f"  Total: {summary['total_subjects']} prompts × ~{summary['total_trials']/summary['total_subjects']:.0f} trials/subj = {summary['total_trials']} trial positions")


if __name__ == "__main__":
    main()
