import tempfile, unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from review_store import ReviewStore, ConflictError


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "review.db"
        self.s = ReviewStore(self.path)
        self.args = (
            "Study",
            "The study investigates a new diagnostic testing method.",
            "version1",
            {"topics": ["Diagnosis"]},
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_concurrent_duplicate_requests_one_document(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(lambda _: self.s.register(*self.args), range(12)))
        self.assertEqual(len({r["id"] for r in rows}), 1)

    def test_stale_review_cannot_overwrite_newer(self):
        row = self.s.register(*self.args)
        self.s.review(row["id"], 0, ["Diagnosis"], [], ["Diagnosis"])
        with self.assertRaises(ConflictError):
            self.s.review(row["id"], 0, [], [], ["Diagnosis"])
        self.assertEqual(self.s.get(row["id"])["review"]["labels"], ["Diagnosis"])

    def test_reopen_retains_review(self):
        row = self.s.register(*self.args)
        self.s.review(row["id"], 0, ["Diagnosis"], [], ["Diagnosis"])
        self.assertEqual(ReviewStore(self.path).get(row["id"])["status"], "reviewed")

    def test_model_version_separate(self):
        a = self.s.register(*self.args)
        b = self.s.register(self.args[0], self.args[1], "version2", self.args[3])
        self.assertNotEqual(a["id"], b["id"])

    def test_fabricated_quote_rejected(self):
        row = self.s.register(*self.args)
        with self.assertRaises(ValueError):
            self.s.review(
                row["id"],
                0,
                ["Diagnosis"],
                [{"label": "Diagnosis", "quote": "Invented source sentence not in document."}],
                ["Diagnosis"],
            )

    def test_exact_quote_accepted(self):
        row = self.s.register(*self.args)
        r = self.s.review(
            row["id"],
            0,
            ["Diagnosis"],
            [{"label": "Diagnosis", "quote": "a new diagnostic testing method"}],
            ["Diagnosis"],
        )
        self.assertEqual(r["revision"], 1)

    def test_unknown_label_rejected(self):
        row = self.s.register(*self.args)
        with self.assertRaises(ValueError):
            self.s.review(row["id"], 0, ["MadeUp"], [], ["Diagnosis"])

    def test_llm_runs_do_not_overwrite_human_review(self):
        row = self.s.register(*self.args)
        reviewed = self.s.review(row["id"], 0, ["Diagnosis"], [], ["Diagnosis"])
        self.s.append_llm_run(row["id"], {"verified_topics": ["Treatment"]})
        after = self.s.get(row["id"])
        self.assertEqual(after["review"], reviewed["review"])
        self.assertEqual(after["revision"], reviewed["revision"])
        self.assertEqual(len(self.s.llm_runs(row["id"])), 1)


if __name__ == "__main__":
    unittest.main()
