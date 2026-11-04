"""Contract checks against the actual pinned BERT tokenizer, no model download."""

import unittest
from bag_model import ROOT, bert_chunk
from transformers import AutoTokenizer


class TokenWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(ROOT / "models/biomedbert", local_files_only=True)

    def test_native_single_sequence_layout(self):
        for text in [
            "COVID-19 diagnosis and treatment.",
            "SARS-CoV-2 vaccination: a case report.",
            "",
        ]:
            ids = self.tok.encode(text, add_special_tokens=False)
            self.assertEqual(
                bert_chunk(ids, self.tok), self.tok.encode(text, add_special_tokens=True)
            )

    def test_full_window_and_padding_mask(self):
        ids = self.tok.encode("viral treatment " * 150, add_special_tokens=False)
        full = bert_chunk(ids[:126], self.tok)
        short = bert_chunk(ids[:7], self.tok)
        self.assertEqual(len(full), 128)
        x = self.tok.pad(
            [{"input_ids": a, "attention_mask": [1] * len(a)} for a in [full, short]],
            padding=True,
            return_tensors="pt",
        )
        self.assertEqual(tuple(x["input_ids"].shape), (2, 128))
        self.assertEqual(int(x["attention_mask"][1].sum()), len(short))
        self.assertEqual(int(x["input_ids"][1, len(short) - 1]), self.tok.sep_token_id)
        self.assertTrue((x["attention_mask"][1, len(short) :] == 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
