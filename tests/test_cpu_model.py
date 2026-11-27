import importlib.util
import tempfile
import unittest
from pathlib import Path

from myvllm import BatchItem, Engine, Phase


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                         ("torch", "transformers", "safetensors")), "install the model extra")
class CPUModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        from safetensors.torch import save_file
        from myvllm.cpu_runner import load_text_model

        torch.manual_seed(42)
        cfg = Qwen3_5TextConfig(
            vocab_size=96, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=4,
            linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
            pad_token_id=0, eos_token_id=2, tie_word_embeddings=False,
            rope_parameters={"rope_theta": 10000000.0, "rope_type": "default",
                             "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
        )
        cfg._attn_implementation = "eager"
        cls.ref = Qwen3_5ForCausalLM(cfg).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            cfg.save_pretrained(path)
            tensors = {}
            for k, value in cls.ref.state_dict().items():
                key = k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k
                tensors[key] = value
            tensors["model.visual.unused"] = torch.zeros(1)
            save_file(tensors, path / "tiny.safetensors")
            cls.model = load_text_model(path)

    def test_logits_across_chunks_match_full_reference(self):
        import torch
        tokens = torch.tensor([7, 13, 5, 42, 18, 11, 17])
        with torch.inference_mode():
            expected = self.ref(tokens[None], use_cache=False).logits[0]
            states = ()
            for start, end in [(0, 2), (2, 5), (5, 6), (6, 7)]:
                hidden, states = self.model(tokens[start:end], start, states)
                torch.testing.assert_close(self.model.lm_head(hidden), expected[start:end],
                                           atol=2e-5, rtol=2e-4)
            self.assertEqual(states[0].memory.dtype, torch.float32)
            self.assertEqual(states[1].keys.shape[1], len(tokens))

    def test_engine_isolation_release_and_failed_batch_retry(self):
        from myvllm.cpu_runner import CPURunner
        runner = CPURunner(self.model)
        first = BatchItem(0, Phase.PREFILL, 0, (1, 2), True)
        invalid = BatchItem(1, Phase.DECODE, 4, (6,), True)
        with self.assertRaises(ValueError):
            runner.execute((first, invalid))
        self.assertEqual(runner._requests, {})
        engine = Engine(runner, token_budget=3)
        prompts = [[7, 12], [5, 10, 16]]
        outputs = {}
        for prompt in prompts:
            outputs[engine.add_request(prompt, max_new_tokens=3)] = []
        while engine.has_requests:
            for event in engine.step():
                outputs[event.request_id].append(event.token_id)
        self.assertEqual(runner._requests, {})
        import torch
        with torch.inference_mode():
            for idx, prompt in enumerate(prompts):
                context = list(prompt)
                for token in outputs[idx]:
                    logits = self.ref(torch.tensor([context]), use_cache=False).logits
                    self.assertEqual(token, logits[0, -1].argmax().item())
                    context.append(token)
