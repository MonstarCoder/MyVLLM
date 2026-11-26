import unittest

from myvllm import Engine, Phase, TokenEvent


class RecordingRunner:
    def __init__(self):
        self.batches = []

    def execute(self, batch):
        self.batches.append(batch)
        return {item.request_id: item.token_ids[-1] + 1 for item in batch if item.sample}


class EngineTest(unittest.TestCase):
    def test_chunk_boundary_and_first_decode_position(self):
        runner = RecordingRunner()
        engine = Engine(runner, token_budget=2)
        req = engine.add_request([3, 4, 5], max_new_tokens=2)
        self.assertEqual(engine.step(), [])
        self.assertTrue(engine.has_requests)
        self.assertEqual(engine.step(), [TokenEvent(req, 6, False)])
        self.assertEqual(engine.step(), [TokenEvent(req, 7, True)])
        self.assertEqual(engine.step(), [])
        self.assertFalse(engine.has_requests)
        self.assertEqual(
            [(b[0].phase, b[0].position, b[0].token_ids, b[0].sample) for b in runner.batches],
            [(Phase.PREFILL, 0, (3, 4), False), (Phase.PREFILL, 2, (5,), True),
             (Phase.DECODE, 3, (6,), True)],
        )

    def test_interleaved_arrivals_and_eos(self):
        runner = RecordingRunner()
        engine = Engine(runner, token_budget=4, eos_ids=[9])
        first = engine.add_request([7], max_new_tokens=4)
        second = engine.add_request([8], max_new_tokens=4)
        self.assertEqual(engine.step(), [TokenEvent(first, 8, False), TokenEvent(second, 9, True)])
        third = engine.add_request([1], max_new_tokens=1)
        self.assertEqual(engine.step(), [TokenEvent(first, 9, True)])
        self.assertEqual(engine.step(), [TokenEvent(third, 2, True)])
        self.assertFalse(engine.has_requests)

    def test_only_finished_prompt_in_shared_batch_gets_a_token(self):
        runner = RecordingRunner()
        engine = Engine(runner, token_budget=3)
        first = engine.add_request([1, 2], max_new_tokens=1)
        second = engine.add_request([3, 4], max_new_tokens=1)
        self.assertEqual(engine.step(), [TokenEvent(first, 3, True)])
        self.assertEqual([item.token_ids for item in runner.batches[0]], [(1, 2), (3,)])
        self.assertEqual(engine.step(), [TokenEvent(second, 5, True)])
        self.assertEqual(runner.batches[1][0].position, 1)

    def test_invalid_result_leaves_host_progress_unchanged(self):
        class BadFirstReply(RecordingRunner):
            def execute(self, batch):
                result = super().execute(batch)
                return {} if len(self.batches) == 1 else result

        runner = BadFirstReply()
        engine = Engine(runner, token_budget=1)
        req = engine.add_request([4], max_new_tokens=1)
        with self.assertRaises(ValueError):
            engine.step()
        self.assertEqual(engine.step(), [TokenEvent(req, 5, True)])
        self.assertEqual(runner.batches[0], runner.batches[1])


if __name__ == "__main__":
    unittest.main()
