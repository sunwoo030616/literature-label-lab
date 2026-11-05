import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evidence import validate, verify_quote, visible_document


class EvidenceTests(unittest.TestCase):
    def test_full_original_quote_outside_visible_is_rejected(self):
        row = dict(title="T", abstract="x" * 1300 + "Late invisible evidence sentence.")
        docs = {"1": visible_document(row)}
        self.assertFalse(
            verify_quote("1", "Late invisible evidence sentence.", docs)["valid"]
        )
        self.assertFalse(verify_quote("2", "x" * 20, docs)["valid"])
        self.assertTrue(verify_quote("1", "x" * 20, docs)["valid"])

    def test_wrong_types_and_schema_fail_closed(self):
        for value in [
            "not JSON",
            "[]",
            '{"answer":42,"citations":[]}',
            '{"answer":"","citations":[],"extra":1}',
        ]:
            self.assertFalse(validate(value, {})["schema_valid"])
        self.assertFalse(verify_quote([], "some quotation here", {})["valid"])

    def test_mixed_citations_do_not_certify_generated_answer(self):
        text = "A valid verbatim quotation on a specific topic."
        result = validate(
            json.dumps(
                dict(
                    answer="An unsupported generated claim.",
                    citations=[dict(pmid="1", quote=text), dict(pmid="2", quote=text)],
                )
            ),
            {"1": text},
        )
        self.assertEqual(len(result["verified_excerpts"]), 1)
        self.assertFalse(result["proposed_answer_semantically_verified"])
        self.assertNotIn("answer", result["user_response"])

    def test_empty_answer_abstains_and_fences_are_not_silently_repaired(self):
        self.assertEqual(
            validate('{"answer":"","citations":[]}', {})["status"], "abstained"
        )
        self.assertFalse(validate("```json\n{}\n```", {})["schema_valid"])


if __name__ == "__main__":
    unittest.main()
