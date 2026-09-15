#!/usr/bin/env python3
"""Anexo A, apartado A.2 · Adaptadores: configuración común de entrenamiento.

Prepara cada lote de entrenamiento marcando qué posiciones de la secuencia cuentan para la pérdida:
las que caen entre los marcadores de elección reciben su etiqueta y todas las demás se excluyen del gradiente. Es la
convención del entrenamiento original de Centaur y la comparten todos los adaptadores del estudio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Lazy imports: only require torch/transformers when actually using
# the DataCollator/compute_answer_token_nll (training/eval paths).
# The pure Python span-finding logic can be smoke-tested without GPU stack.
try:
    import torch
    from transformers import PreTrainedTokenizerBase
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    PreTrainedTokenizerBase = object  # type alias for type hints in non-GPU env


IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss default ignore_index


def find_answer_spans(input_ids: list[int], tokenizer: PreTrainedTokenizerBase,
                     open_marker: str = "<<", close_marker: str = ">>") -> list[tuple[int, int]]:
    """Identify token spans between `<<` and `>>` markers.

    Returns list of (start_idx, end_idx) — half-open intervals over input_ids.
    Token indices INSIDE the markers (i.e. the answer content) are returned.
    Marker tokens themselves are NOT included (they remain non-answer
    contextual tokens).

    Implementation: decode token-by-token searching for marker substrings.
    Two-pass approach to handle marker tokenization variability across
    different tokenizers / BPE merges.

    Args:
        input_ids: tokenized input as list of ints.
        tokenizer: HuggingFace tokenizer (used to decode token by token).
        open_marker: opening marker string (default "<<").
        close_marker: closing marker string (default ">>").
    """
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    spans_in_text: list[tuple[int, int]] = []
    pos = 0
    while True:
        i_open = decoded.find(open_marker, pos)
        if i_open == -1:
            break
        i_after_open = i_open + len(open_marker)
        i_close = decoded.find(close_marker, i_after_open)
        if i_close == -1:
            break
        spans_in_text.append((i_after_open, i_close))
        pos = i_close + len(close_marker)

    if not spans_in_text:
        return []

    # Map text-char spans to token-index spans.
    # We re-encode and use offset_mapping if available (fast tokenizers).
    # Fallback: incremental decode (slow but reliable).
    spans_in_tokens: list[tuple[int, int]] = []

    if hasattr(tokenizer, "is_fast") and tokenizer.is_fast:
        encoding = tokenizer(decoded, return_offsets_mapping=True,
                            add_special_tokens=False, return_attention_mask=False)
        offsets = encoding["offset_mapping"]
        # We need to align our `input_ids` to `encoding["input_ids"]`. They
        # may differ if special tokens were added in original tokenization.
        # Conservative: find the first index where they align and use offsets.
        encoded_ids = encoding["input_ids"]
        # Find anchor: try matching the full input_ids inside encoded_ids
        anchor = -1
        for k in range(0, max(1, len(encoded_ids) - len(input_ids) + 1)):
            if encoded_ids[k:k + len(input_ids)] == list(input_ids):
                anchor = k
                break
        if anchor < 0:
            # Fallback to char-by-char decode mapping (below)
            return _spans_by_decoding(input_ids, tokenizer, spans_in_text)

        # [revisión interna] O9.N1: Use OVERLAP semantics (matching slow path) instead
        # of `a >= cs`. Without this, fast tokenizers that emit a merged
        # marker+answer token (e.g. "<<J" as a single token) silently drop
        # the answer span because the token's offset start `a` is BEFORE the
        # char-span start `cs` (which is post-marker). Result: zero answer
        # labels → all-ignored training loss → silent corruption.
        #
        # Overlap test (consistent with _spans_by_decoding):
        #   token's char-range [a, b) overlaps with answer char-range [cs, ce)
        #   iff (b > cs) AND (a < ce). Drop tokens that ONLY contain the
        #   marker chars (heuristic: token decoded text is exactly the marker
        #   or substring of marker; we accept marker-merged tokens because
        #   answer characters are inside that token too).
        sub_offsets = offsets[anchor:anchor + len(input_ids)]
        for (cs, ce) in spans_in_text:
            tok_start, tok_end = None, None
            for t_idx, (a, b) in enumerate(sub_offsets):
                if b is None or a is None:
                    continue
                # Overlap test: token [a, b) ∩ answer [cs, ce) is non-empty
                if b > cs and a < ce:
                    if tok_start is None:
                        tok_start = t_idx
                    tok_end = t_idx + 1
            if tok_start is not None and tok_end is not None and tok_end > tok_start:
                spans_in_tokens.append((tok_start, tok_end))
        return spans_in_tokens

    # Slow path
    return _spans_by_decoding(input_ids, tokenizer, spans_in_text)


def _spans_by_decoding(input_ids: list[int], tokenizer: PreTrainedTokenizerBase,
                      spans_in_text: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Slow path: incremental decoding to map char-spans to token-spans."""
    spans_in_tokens: list[tuple[int, int]] = []
    # Build cumulative char lengths per token via incremental decode
    char_starts = [0]
    running = ""
    for tok in input_ids:
        piece = tokenizer.decode([tok], skip_special_tokens=False)
        running += piece
        char_starts.append(len(running))
    # char_starts[i] = char index where token i begins in the running decode
    for (cs, ce) in spans_in_text:
        tok_start, tok_end = None, None
        for i in range(len(input_ids)):
            tok_cs = char_starts[i]
            tok_ce = char_starts[i + 1]
            # Overlap test
            if tok_ce <= cs or tok_cs >= ce:
                continue
            if tok_start is None:
                tok_start = i
            tok_end = i + 1
        if tok_start is not None and tok_end is not None and tok_end > tok_start:
            spans_in_tokens.append((tok_start, tok_end))
    return spans_in_tokens


def build_answer_mask_labels(input_ids: list[int], tokenizer: PreTrainedTokenizerBase,
                            open_marker: str = "<<",
                            close_marker: str = ">>") -> list[int]:
    """Return labels where non-answer tokens are set to IGNORE_INDEX (-100).

    Convention: labels[i] = input_ids[i] if i is an answer token, else -100.
    Causal LM trainers compute loss over (logits[i-1], labels[i]) where
    labels[i] = -100 is ignored.

    For autoregressive prediction we want loss[i] to fire when predicting
    the answer token at position i. The standard HF causal LM contract is
    labels = input_ids shifted internally, so setting labels[i] = -100
    for non-answer positions causes those positions to be ignored.
    """
    spans = find_answer_spans(input_ids, tokenizer, open_marker, close_marker)
    labels = [IGNORE_INDEX] * len(input_ids)
    for (start, end) in spans:
        for i in range(start, end):
            labels[i] = input_ids[i]
    # [revisión interna] O9.N1: silent-loss guard. If a prompt has marker pairs in
    # text but the tokenizer produces zero answer tokens, the batch's loss
    # would degenerate to NaN/zero with no signal. Surface this loudly.
    n_answer = sum(1 for x in labels if x != IGNORE_INDEX)
    if spans and n_answer == 0:
        raise ValueError(
            f"masked_loss_collator: text contains {len(spans)} marker pair(s) "
            f"but tokenizer produced 0 answer tokens after span mapping. "
            f"This indicates a tokenizer edge case (merged marker+answer "
            f"tokens, exotic vocab, etc.). Inspect tokenizer behavior for "
            f"`<<X>>` patterns before launching."
        )
    return labels


def _require_torch():
    if not _HAS_TORCH:
        raise ImportError(
            "masked_loss_collator: AnswerTokenMaskedDataCollator + "
            "compute_answer_token_nll require torch + transformers. "
            "Install them in the launch environment (Lightning GPU Studio)."
        )


@dataclass
class AnswerTokenMaskedDataCollator:
    """DataCollator that builds answer-token masked labels.

    Drop-in replacement for `DataCollatorForLanguageModeling(mlm=False)` that
    implements the Centaur-paper-spec masked loss contract.

    Usage:
        collator = AnswerTokenMaskedDataCollator(tokenizer=tokenizer)
        trainer = Trainer(..., data_collator=collator)
    """
    tokenizer: PreTrainedTokenizerBase
    open_marker: str = "<<"
    close_marker: str = ">>"
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, Any]]):
        _require_torch()
        # `features` is a list of {"input_ids": [...], "attention_mask": [...]} dicts
        # produced by `tokenize_fn` upstream.

        all_input_ids = []
        all_labels = []
        max_len = 0

        for feat in features:
            input_ids = list(feat["input_ids"])
            labels = build_answer_mask_labels(
                input_ids, self.tokenizer, self.open_marker, self.close_marker,
            )
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            max_len = max(max_len, len(input_ids))

        if self.pad_to_multiple_of is not None:
            rem = max_len % self.pad_to_multiple_of
            if rem != 0:
                max_len += (self.pad_to_multiple_of - rem)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        # Right-pad to max_len
        batch_input_ids = []
        batch_labels = []
        batch_attention = []
        for input_ids, labels in zip(all_input_ids, all_labels):
            pad = max_len - len(input_ids)
            batch_input_ids.append(input_ids + [pad_id] * pad)
            batch_labels.append(labels + [IGNORE_INDEX] * pad)
            batch_attention.append([1] * len(input_ids) + [0] * pad)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def compute_answer_token_nll(model, tokenizer, text: str, device=None,
                              open_marker: str = "<<",
                              close_marker: str = ">>",
                              max_length: int | None = None) -> dict[str, Any]:
    _require_torch()
    """Compute token-weighted mean NLL over answer tokens only.

    Replaces the full-sequence NLL convention from the old eval scripts.
    Used by eval_2way.py and eval_3way.py after [revisión interna] O8.N3.

    [revisión interna] O12.N5: opt-in `max_length` truncation para evitar OOM en VRAM
    constrained envs (A100 40GB Colab). Si max_length is None (default),
    tokenize without truncation (canonical behavior). Truncation may drop
    `<<X>>` answer spans at the tail; caller responsible for picking
    max_length high enough to retain all spans.

    Returns:
        {
            "n_answer_tokens": int,
            "sum_nll": float,
            "mean_nll": float,
            "n_answer_spans": int,
        }
    """
    if device is None:
        device = next(model.parameters()).device

    if max_length is not None:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    else:
        enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0].tolist()
    labels = build_answer_mask_labels(input_ids, tokenizer, open_marker, close_marker)

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels_t = torch.tensor([labels], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids_t, labels=labels_t)
        # HF causal LM loss is mean NLL over non-ignored labels.
        # outputs.loss already gives the correct mean over answer tokens.
        mean_nll = float(outputs.loss.item())

    n_answer = sum(1 for lab in labels if lab != IGNORE_INDEX)
    # spans count
    spans = find_answer_spans(input_ids, tokenizer, open_marker, close_marker)
    return {
        "n_answer_tokens": n_answer,
        "sum_nll": mean_nll * n_answer if n_answer > 0 else 0.0,
        "mean_nll": mean_nll if n_answer > 0 else float("nan"),
        "n_answer_spans": len(spans),
    }


def compute_per_trial_logprobs(model, tokenizer, text: str,
                                top_k: int = 10,
                                max_length: int | None = None,
                                device=None) -> dict[str, Any]:
    """Per-trial logprobs + top-K alternatives + aggregate NLL en single forward pass.

    Ported from Paper 1 Lightning notebook _per_trial_logprobs implementation.
    Per cada answer span `<<X>>` en el texto, captura:
      - target_tokens + target_token_logprobs (logprob de la acción humana actual)
      - model_argmax_token + model_argmax_logprob (qué eligió el modelo)
      - top_k_tokens + top_k_logprobs (top-K alternativas ranked at decision pos)

    Args:
        model: HF CausalLM model (e.g., Llama + Centaur adapter).
        tokenizer: HF tokenizer (FAST tokenizer required; needs offset_mapping).
        text: input prompt con spans `<<X>>` para los trials.
        top_k: K alternativas a guardar en decision position de cada span (default 10).
        max_length: opt-in truncation (default None = no truncation).
        device: torch device (auto-detect from model si None).

    Returns:
        {
            "aggregate_nll": float,         # mean NLL across all answer tokens (= C1+C2 metric)
            "n_answer_tokens": int,
            "n_answer_spans": int,
            "per_trial": [                  # one entry per trial (span)
                {
                    "trial_index": int,
                    "target_text": str,     # raw text between <<>>
                    "start_char": int,
                    "end_char": int,
                    "target_tokens": [str], # decoded answer tokens
                    "target_token_logprobs": [float],
                    "decision_position": int,
                    "model_argmax_token_id": int,
                    "model_argmax_token": str,
                    "model_argmax_logprob": float,
                    "top_k": int,
                    "top_k_token_ids": [int],
                    "top_k_tokens": [str],
                    "top_k_logprobs": [float],
                },
                ...
            ]
        }
    """
    _require_torch()
    import re

    if device is None:
        device = next(model.parameters()).device

    # 1. Find char spans entre <<X>> markers via regex
    spans_meta = [
        {
            "trial_index": idx,
            "target_text": m.group(1).strip(),
            "start_char": m.start(1),
            "end_char": m.end(1),
        }
        for idx, m in enumerate(re.finditer(r"<<\s*(.*?)\s*>>", text, flags=re.DOTALL))
    ]

    if not spans_meta:
        return {
            "aggregate_nll": float("nan"),
            "n_answer_tokens": 0,
            "n_answer_spans": 0,
            "per_trial": [],
        }

    # 2. Tokenize with offset mapping (REQUIRES fast tokenizer)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "compute_per_trial_logprobs requires a FAST tokenizer "
            "(offset_mapping). Llama-3.1 tokenizer is fast by default."
        )

    enc_kwargs = dict(return_tensors="pt", return_offsets_mapping=True)
    if max_length is not None:
        enc_kwargs.update(truncation=True, max_length=max_length)
    enc = tokenizer(text, **enc_kwargs)
    offset_mapping = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    # 3. Forward pass with log_softmax over vocab dim
    with torch.no_grad():
        outputs = model(**enc)
    log_probs = torch.log_softmax(outputs.logits[0], dim=-1)
    input_ids = enc["input_ids"][0]

    # 4. For each span, map char offsets to token positions + extract logprobs
    per_trial = []
    sum_logprobs = 0.0
    n_tokens_total = 0
    for span in spans_meta:
        token_positions = [
            pos
            for pos, (start, end) in enumerate(offset_mapping)
            if end > span["start_char"] and start < span["end_char"] and pos > 0
        ]
        token_logprobs = []
        target_tokens = []
        for pos in token_positions:
            token_id = int(input_ids[pos])
            target_tokens.append(tokenizer.decode([token_id], skip_special_tokens=False))
            lp = float(log_probs[pos - 1, token_id].detach().cpu())
            token_logprobs.append(lp)
            sum_logprobs += lp
            n_tokens_total += 1

        entry = {
            **span,
            "target_tokens": target_tokens,
            "target_token_logprobs": token_logprobs,
        }

        # top-K alternatives at decision position (first answer token of this span)
        if token_positions and top_k and int(top_k) > 0:
            first_pos = token_positions[0]
            dist = log_probs[first_pos - 1]
            argmax_id = int(dist.argmax().item())
            argmax_logprob = float(dist[argmax_id].detach().cpu())
            k = min(int(top_k), int(dist.shape[-1]))
            topk_vals, topk_ids = torch.topk(dist, k=k)
            topk_ids_list = [int(x) for x in topk_ids.detach().cpu().tolist()]
            topk_logprobs_list = [float(x) for x in topk_vals.detach().cpu().tolist()]
            entry.update({
                "decision_position": int(first_pos),
                "model_argmax_token_id": argmax_id,
                "model_argmax_token": tokenizer.decode([argmax_id], skip_special_tokens=False),
                "model_argmax_logprob": argmax_logprob,
                "top_k": int(k),
                "top_k_token_ids": topk_ids_list,
                "top_k_tokens": [tokenizer.decode([tid], skip_special_tokens=False) for tid in topk_ids_list],
                "top_k_logprobs": topk_logprobs_list,
            })

        per_trial.append(entry)

    # 5. Aggregate NLL = mean negative log-likelihood across all answer tokens
    if n_tokens_total > 0:
        aggregate_nll = -sum_logprobs / n_tokens_total
    else:
        aggregate_nll = float("nan")

    return {
        "aggregate_nll": aggregate_nll,
        "n_answer_tokens": n_tokens_total,
        "n_answer_spans": len(spans_meta),
        "per_trial": per_trial,
    }


# Smoke test (run as: python masked_loss_collator.py)
if __name__ == "__main__":
    print("=" * 60)
    print("masked_loss_collator smoke test")
    print("=" * 60)

    # ===========================================================
    # [revisión interna] O9.N1 test fixtures:
    # 3 mock scenarios cubriendo el comportamiento real de Llama tokenizer.
    # Cada fixture proporciona (input_ids, decode_per_token, decode_full)
    # para que find_answer_spans pueda computar spans sin requerir HF.
    # ===========================================================

    class _FixtureTokenizer:
        """Mock tokenizer con per-token + full decode predefinidos.

        scenario_id ∈ {separated, merged_marker_left, merged_marker_right,
                       fully_merged, multi_span}.
        """
        is_fast = False  # use slow path (incremental decode)

        def __init__(self, per_token_decoded: list[str]):
            self.per_token = per_token_decoded
            self.pad_token_id = 0
            self.eos_token_id = 1

        def decode(self, ids, **kwargs):
            if not isinstance(ids, (list, tuple)):
                ids = [ids]
            if len(ids) == 1:
                # Per-token decode: ids[0] is index into self.per_token
                idx = ids[0]
                if 0 <= idx < len(self.per_token):
                    return self.per_token[idx]
                return ""
            # Full-sequence decode: concatenate
            return "".join(self.per_token[i] for i in ids if 0 <= i < len(self.per_token))

    fixtures = {
        "separated": {
            "tokens": ["You", " press", " <<", "J", ">>", ". Weather", " is", " fine."],
            "expected_n_answer_tokens": 1,  # only "J"
            "description": "Tokens separados: <<, J, >> son 3 tokens distintos",
        },
        "merged_marker_left": {
            "tokens": ["You", " press", " <<J", ">>", ". Weather"],
            "expected_n_answer_tokens": 1,  # "<<J" (overlap-semantics; marker+answer en un token)
            "description": "Fast tokenizer merge: << + J en un solo token '<<J'",
        },
        "merged_marker_right": {
            "tokens": ["You", " press", " <<", "J>>", ". Weather"],
            "expected_n_answer_tokens": 1,  # "J>>" (overlap-semantics)
            "description": "Fast tokenizer merge: J + >> en un solo token 'J>>'",
        },
        "fully_merged": {
            "tokens": ["You", " press", " <<J>>", ". Weather"],
            "expected_n_answer_tokens": 1,  # "<<J>>" — todo en un token
            "description": "Tokenizer extremo: <<J>> en un solo token",
        },
        "multi_span": {
            "tokens": ["You", " press", " <<", "A", ">>", ". You press", " <<", "B", ">>", "."],
            "expected_n_answer_tokens": 2,
            "description": "Múltiples answer spans separados",
        },
    }

    all_ok = True
    for fname, fdata in fixtures.items():
        tokens = fdata["tokens"]
        tokenizer = _FixtureTokenizer(tokens)
        # input_ids are just indices
        input_ids = list(range(len(tokens)))
        spans = find_answer_spans(input_ids, tokenizer)
        n_answer = sum(end - start for (start, end) in spans)
        expected = fdata["expected_n_answer_tokens"]
        status = "PASS" if n_answer == expected else "FAIL"
        if n_answer != expected:
            all_ok = False
        print(f"[{status}] {fname}: {fdata['description']}")
        print(f"        tokens={tokens}")
        print(f"        spans={spans}, n_answer_tokens={n_answer} (expected {expected})")
        print()

    if all_ok:
        print("OK: all fixtures pass (slow-path overlap semantics correct).")
        print("    Fast-path uses same overlap predicate; recommend real")
        print("    Llama tokenizer integration test before Stage 7.5.x launch.")
    else:
        print("FAIL: fixture(s) above did not pass. Investigate before launch.")
        import sys
        sys.exit(1)
