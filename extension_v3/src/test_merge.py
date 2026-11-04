import unittest, sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "followup_v2/src"))
import torch
from finetune import LoRALinear
from lora_serving import merge_for_inference


class MergeTests(unittest.TestCase):
    def toy(self, training=False):
        layers = []
        for _ in range(12):
            q = LoRALinear(torch.nn.Linear(5, 3), rank=2, alpha=4).eval()
            v = LoRALinear(torch.nn.Linear(5, 3), rank=2, alpha=4).eval()
            with torch.no_grad():
                q.B.normal_()
                v.B.normal_()
            layers.append(
                SimpleNamespace(attention=SimpleNamespace(self=SimpleNamespace(query=q, value=v)))
            )
        return SimpleNamespace(
            training=training, encoder=SimpleNamespace(encoder=SimpleNamespace(layer=layers))
        )

    def test_merged_outputs_and_idempotence(self):
        torch.manual_seed(17)
        m = self.toy()
        x = torch.randn(7, 5)
        before = [l.attention.self.query(x).detach() for l in m.encoder.encoder.layer]
        self.assertEqual(merge_for_inference(m), 24)
        for layer, expected in zip(m.encoder.encoder.layer, before):
            self.assertTrue(torch.allclose(layer.attention.self.query(x), expected, atol=1e-6))
        self.assertEqual(merge_for_inference(m), 0)

    def test_refuses_training_mode(self):
        with self.assertRaises(ValueError):
            merge_for_inference(self.toy(training=True))


if __name__ == "__main__":
    unittest.main()
