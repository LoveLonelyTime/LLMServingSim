"""AzureLLMInferenceTrace CSV -> LLMServingSim JSONL.

Converts an Azure Functions / LLM-inference trace CSV with columns
``TIMESTAMP, ContextTokens, GeneratedTokens`` into the flat-request JSONL
consumed by ``python -m serving`` and ``python -m bench``::

    {
      "input_toks":      <int>,   # ContextTokens (prompt length)
      "output_toks":     <int>,   # ContextTokens + GeneratedTokens (total length)
      "arrival_time_ns": <int>,   # parsed from TIMESTAMP, aligned so first == 0
      "input_tok_ids":   [...],   # left empty (not used by perf simulation)
      "output_tok_ids":  [...]
    }

``output_toks`` is the *total* token target (prompt + generated), matching
the simulator's ``num_tokens_reached`` semantics — the same convention
``example_trace.jsonl`` uses. ``input_tok_ids``/``output_tok_ids`` are not
used by performance simulation (only prefix-caching hash inputs would be),
so they are emitted as empty lists.

TIMESTAMP is a naive wall-clock string (e.g. ``2023-11-16 18:17:03.9799600``,
7 fractional digits). Arrival times are converted to ns and, by default,
aligned so the first request arrives at ``arrival_time_ns = 0``
(``--no-align-first`` keeps the raw epoch-relative times).

Usage:
    python -m workloads.generators azure-code \
        --input workloads/AzureLLMInferenceTrace_code.csv \
        --output workloads/azure_code.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

_EPOCH = datetime(1970, 1, 1)


# ---------------------------------------------------------------------------
# CLI plumbing — invoked from workloads.generators.__main__
# ---------------------------------------------------------------------------

def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", default="workloads/AzureLLMInferenceTrace_code.csv",
                   help="Source CSV (columns TIMESTAMP, ContextTokens, "
                        "GeneratedTokens). Default: "
                        "workloads/AzureLLMInferenceTrace_code.csv")
    p.add_argument("--output", required=True,
                   help="Output JSONL path.")
    p.add_argument("--align-first", action=argparse.BooleanOptionalAction,
                   dest="align_first", default=True,
                   help="Align arrival so the first request is "
                        "arrival_time_ns=0 (default). Use --no-align-first to "
                        "keep raw epoch-relative timestamps.")
    p.add_argument("--first-arrival-sec", type=float, default=0.0,
                   dest="first_arrival_sec",
                   help="Extra offset (seconds) added to the first request's "
                        "arrival. Default 0.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _parse_timestamp(text: str) -> datetime:
    """Parse a naive wall-clock timestamp with arbitrary fractional digits."""
    ts = text.strip()
    # fromisoformat handles 1-6 fractional digits in all versions; pad/truncate
    # a longer fraction to 6 for maximal compatibility.
    if "." in ts:
        head, frac = ts.split(".", 1)
        frac = (frac + "000000")[:6]
        ts = f"{head}.{frac}"
    return datetime.fromisoformat(ts)


def run(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    # utf-8-sig: strip a possible leading BOM from Excel-exported CSVs.
    rows = []
    skipped = 0
    with in_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            try:
                ts = _parse_timestamp(raw["TIMESTAMP"])
                ctx = int(raw["ContextTokens"])
                gen = int(raw["GeneratedTokens"])
            except (KeyError, ValueError) as exc:
                skipped += 1
                continue
            if ctx < 0 or gen < 0:
                skipped += 1
                continue
            ns = int((ts - _EPOCH).total_seconds() * 1_000_000_000)
            rows.append({"ns": ns, "ctx": ctx, "gen": gen})

    rows.sort(key=lambda r: r["ns"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_ns = int(args.first_arrival_sec * 1_000_000_000)
    if args.align_first and rows:
        base_ns -= rows[0]["ns"]

    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for r in rows:
            arrival = r["ns"] + base_ns
            if arrival < 0:
                arrival = 0
            row = {
                "input_toks": r["ctx"],
                "output_toks": r["ctx"] + r["gen"],
                "arrival_time_ns": arrival,
                "input_tok_ids": [],
                "output_tok_ids": [],
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    span_s = (rows[-1]["ns"] - rows[0]["ns"]) / 1e9 if rows else 0.0
    print(f"Wrote {written} requests -> {out_path}")
    print(f"  (skipped {skipped} malformed rows, {span_s:.1f}s arrival span "
          f"after alignment)")
    return 0
