"""Greedy text completion with the eager CPU backend."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .cpu_runner import CPURunner, load_text_model
from .engine import Engine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local Qwen3.8-27B checkpoint directory")
    parser.add_argument("prompt", help="raw text completion prompt")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--token-budget", type=int, default=64)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.token_budget < 1:
        parser.error("--max-new-tokens and --token-budget must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = tokenizer.encode(args.prompt)
    if not tokens:
        parser.error("prompt must encode to at least one token")
    config_path = args.model / "generation_config.json"
    settings = json.loads(config_path.read_text()) if config_path.exists() else {}
    eos = settings.get("eos_token_id", tokenizer.eos_token_id)
    eos_ids = [eos] if isinstance(eos, int) else eos or []
    runner = CPURunner(load_text_model(args.model))
    engine = Engine(runner, token_budget=args.token_budget, eos_ids=eos_ids)
    engine.add_request(tokens, max_new_tokens=args.max_new_tokens)
    output = []
    while engine.has_requests:
        output.extend(event.token_id for event in engine.step())
    print(tokenizer.decode(output, skip_special_tokens=True))


if __name__ == "__main__":
    main()
