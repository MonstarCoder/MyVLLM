"""Compare prefill and teacher-forced decode on a local BF16 checkpoint."""
import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from myvllm.cpu_runner import load_text_model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--steps", type=int, default=3, help="first prefill plus subsequent decode steps")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    reference = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager",
    ).eval()
    tokens = tokenizer(args.prompt, return_tensors="pt").input_ids
    cache = None
    samples = []
    for _ in range(args.steps):
        result = reference(input_ids=tokens, past_key_values=cache, use_cache=True,
                           logits_to_keep=1)
        cache = result.past_key_values
        logits = result.logits[0, -1].float().clone()
        samples.append((tokens[0], logits))
        tokens = logits.argmax().reshape(1, 1)
    del reference, result, cache
    gc.collect()

    model = load_text_model(args.model)
    states = ()
    position = 0
    for tokens, expected in samples:
        hidden, states = model(tokens, position, states)
        actual = model.lm_head(hidden[-1:]).squeeze(0).float()
        delta = (actual - expected).abs()
        print(f"position={position} inputs={tokens.tolist()} "
              f"hf={expected.argmax().item()} myvllm={actual.argmax().item()} "
              f"max_abs={delta.max().item():.6f} mean_abs={delta.mean().item():.6f}")
        position += len(tokens)


if __name__ == "__main__":
    main()
