import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quote_recovery import recover_edge_ellipsis


def output(quote):
    return json.dumps(
        dict(answer="a proposed answer", citations=[dict(pmid="1", quote=quote)])
    )


class RecoveryTests(unittest.TestCase):
    def test_recovers_only_unique_contiguous_visible_text(self):
        quote = "in less than 2 min."
        result = recover_edge_ellipsis(
            output("..." + quote), {"1": "This occurred " + quote}
        )
        self.assertEqual(result["verified_excerpts"], [dict(pmid="1", quote=quote)])
        self.assertFalse(result["citations"][0]["valid"])
        self.assertEqual(
            result["recovered_excerpts"][0]["original_quote"], "..." + quote
        )

    def test_internal_ellipsis_fuzzy_and_duplicate_matches_not_repaired(self):
        for quote, doc in [
            (
                "a sufficiently long... altered quotation",
                "a sufficiently long original altered quotation",
            ),
            ("...in less than 2 min.", "in less than 2 min. in less than 2 min."),
            ("...in less than 3 min.", "in less than 2 min."),
        ]:
            result = recover_edge_ellipsis(output(quote), {"1": doc})
            self.assertEqual(result["recovered_excerpts"], [])
            self.assertEqual(result["verified_excerpts"], [])


if __name__ == "__main__":
    unittest.main()
