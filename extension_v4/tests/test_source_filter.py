import tempfile
import unittest
from pathlib import Path
from extension_v4.src.corpus import connect, search


class SourceFilterTests(unittest.TestCase):
    def test_numeric_constraint_is_required_not_just_an_or_search_term(self):
        with tempfile.TemporaryDirectory() as directory:
            db = connect(Path(directory) / "test.sqlite3")
            for pmid, title, abstract in [
                ("1", "Statins COVID study patients", "Trial with 13,449 people."),
                ("2", "Statin meta analysis", "Included 3,449 people."),
            ]:
                db.execute(
                    "INSERT INTO documents VALUES(?,?,?,?,?,?)",
                    (pmid, title, abstract, "", "", pmid),
                )
            db.commit()
            self.assertEqual(
                [r["pmid"] for r in search(db, "statins study patients", 10, ["3449"])],
                [],
            )
            self.assertEqual(
                [r["pmid"] for r in search(db, "statin OR statins", 10, ["3449"])],
                ["2"],
            )
            self.assertEqual(search(db, "statin", 10, ["2026"]), [])
            with self.assertRaises(ValueError):
                search(db, "statin", 10, ["1' OR 1=1"])
            db.close()


if __name__ == "__main__":
    unittest.main()
