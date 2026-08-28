"""CLI dispatch for workload generators.

Usage:
    python -m workloads.generators sharegpt --model <hf-id> --num-reqs 300 --sps 10 \
        --source <path-or-hf-id> --output workloads/sharegpt-<model>-<n>-sps<r>.jsonl
    python -m workloads.generators azure-code \
        --input workloads/AzureLLMInferenceTrace_code.csv --output workloads/azure_code.jsonl
"""

from __future__ import annotations

import argparse
import importlib
import sys

# (subcommand token, module name, help text). Registered in order; a generator
# whose dependencies (e.g. numpy/datasets for sharegpt) are missing is skipped
# with a warning rather than taking down every other subcommand.
_GENERATORS = [
    ("sharegpt", "sharegpt", "ShareGPT -> LLMServingSim JSONL"),
    ("azure-code", "azure_code", "AzureLLMInferenceTrace CSV -> LLMServingSim JSONL"),
]


def _register(sub, token, module, help_text):
    try:
        mod = importlib.import_module(f"workloads.generators.{module}")
        sp = sub.add_parser(token, help=help_text)
        mod.register_args(sp)
        return True
    except Exception as exc:  # missing deps or import error
        sys.stderr.write(f"[warn] generator '{token}' unavailable: {exc}\n")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(prog="workloads.generators")
    sub = parser.add_subparsers(dest="generator", required=True)

    available = {}
    for token, module, help_text in _GENERATORS:
        if _register(sub, token, module, help_text):
            available[token] = module

    args = parser.parse_args()

    module = available[args.generator]
    from importlib import import_module as _imp
    run = _imp(f"workloads.generators.{module}").run
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
