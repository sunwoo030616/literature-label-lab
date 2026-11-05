import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from identifier_guard import preserve_identifiers


class IdentifierTests(unittest.TestCase):
    def test_keeps_numeric_source_anchors_without_mutating_raw_plan(self):
        original = dict(
            tool="search_documents",
            arguments=dict(query="statins COVID-19 patients", limit=3),
        )
        effective, changes = preserve_identifiers(
            original, "3449명을 분석한 statin 연구"
        )
        self.assertIn("3449", effective["arguments"]["query"])
        self.assertNotIn("3449", original["arguments"]["query"])
        self.assertEqual(len(changes), 1)

    def test_explicit_pmid_is_bound_and_unrequested_ids_rejected(self):
        wrong = dict(tool="get_document", arguments=dict(pmid="123"))
        effective, _ = preserve_identifiers(wrong, "Read PMID 32425648")
        self.assertEqual(effective["arguments"]["pmid"], "32425648")
        with self.assertRaises(ValueError):
            preserve_identifiers(wrong, "Find a statin paper")
        with self.assertRaises(ValueError):
            preserve_identifiers(wrong, "Compare PMID 123 and PMID 456")


if __name__ == "__main__":
    unittest.main()
