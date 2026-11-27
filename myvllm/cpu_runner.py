"""Load a local BF16 checkpoint and execute greedy requests on CPU."""
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

from .qwen_text import LayerState, QwenText
from .scheduler import BatchItem


def load_text_model(directory: Path) -> QwenText:
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    cfg = config.get_text_config()
    with torch.device("meta"):
        model = QwenText(cfg)
    expected = dict(model.named_parameters())
    loaded = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as file:
            for name in file.keys():
                if name.startswith("model.language_model."):
                    key = name.removeprefix("model.language_model.")
                elif name == "lm_head.weight":
                    key = name
                else:
                    continue
                if key not in expected or key in loaded:
                    raise ValueError(f"unexpected or duplicate text weight: {name}")
                tensor = file.get_tensor(name)
                if tensor.shape != expected[key].shape or tensor.dtype not in (torch.bfloat16, torch.float32):
                    raise ValueError(f"invalid shape or dtype for {name}")
                loaded[key] = tensor
    if cfg.tie_word_embeddings and "lm_head.weight" not in loaded:
        loaded["lm_head.weight"] = loaded["embed_tokens.weight"]
    model.load_state_dict(loaded, strict=True, assign=True)
    return model.eval().requires_grad_(False)


class CPURunner:
    def __init__(self, model: QwenText):
        self.model = model
        self._requests: dict[int, tuple[int, tuple[LayerState, ...]]] = {}

    @torch.inference_mode()
    def execute(self, batch: tuple[BatchItem, ...]) -> dict[int, int]:
        results = {}
        updates = {}
        for item in batch:
            position, states = self._requests.get(item.request_id, (0, ()))
            if position != item.position:
                raise ValueError(f"cache position mismatch for request {item.request_id}")
            hidden, states = self.model(torch.tensor(item.token_ids), position, states)
            if item.sample:
                results[item.request_id] = int(self.model.lm_head(hidden[-1]).argmax())
            updates[item.request_id] = position + len(item.token_ids), states
        self._requests.update(updates)
        return results

    def release(self, request_id: int) -> None:
        self._requests.pop(request_id, None)
