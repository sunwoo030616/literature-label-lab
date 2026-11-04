import unittest, torch, json
from finetune import LoRALinear
from llm_review import parse


class LearningAndEvidenceTests(unittest.TestCase):
    def test_adapter_initially_preserves_base_and_then_learns(self):
        torch.manual_seed(17)
        base = torch.nn.Linear(5, 3)
        x = torch.randn(6, 5)
        original = base.weight.detach().clone()
        expected = base(x).detach()
        layer = LoRALinear(base, rank=2, alpha=4).eval()
        self.assertTrue(torch.allclose(layer(x), expected))
        opt = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
        loss = layer(x).square().mean()
        loss.backward()
        opt.step()
        self.assertTrue(torch.equal(base.weight, original))
        self.assertGreater(float(layer.B.detach().abs().sum()), 0)
        self.assertFalse(torch.allclose(layer(x), expected))

    def test_verbatim_quote_only(self):
        doc = "This paper describes a diagnostic testing method for COVID-19."
        raw = json.dumps(
            {"topics": [{"label": "Diagnosis", "evidence": "a diagnostic testing method"}]}
        )
        self.assertEqual(parse(raw, doc)["verified_topics"], ["Diagnosis"])

    def test_fabricated_quote_removed(self):
        raw = json.dumps(
            {
                "topics": [
                    {
                        "label": "Diagnosis",
                        "evidence": "Invented diagnostic evidence that is not present.",
                    }
                ]
            }
        )
        p = parse(raw, "This source does not contain the requested quotation.")
        self.assertEqual(p["raw_topics"], ["Diagnosis"])
        self.assertEqual(p["verified_topics"], [])

    def test_invalid_json_is_failure(self):
        self.assertFalse(parse("not json", "document")["valid_json"])

    def test_unknown_topic_never_enters_prediction(self):
        raw = json.dumps(
            {"topics": [{"label": "MadeUp", "evidence": "a long enough real quotation"}]}
        )
        self.assertEqual(parse(raw, "a long enough real quotation")["verified_topics"], [])


if __name__ == "__main__":
    unittest.main()
