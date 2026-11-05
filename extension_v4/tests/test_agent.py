import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent import validate_plan


class PlanTests(unittest.TestCase):
    def test_read_only_plan(self):
        for plan in [
            dict(tool="get_document", arguments=dict(pmid="32425648")),
            dict(
                tool="search_documents",
                arguments=dict(query="serological ELISA", limit=3),
            ),
        ]:
            self.assertEqual(validate_plan(json.dumps(plan)), plan)

    def test_write_tools_bad_ids_and_extra_parameters_rejected(self):
        plans = [
            dict(tool="execute_sql", arguments=dict(sql="DELETE FROM documents")),
            dict(tool="get_document", arguments=dict(pmid="1' OR 1=1")),
            dict(tool="search_documents", arguments=dict(query="x", limit=50)),
            dict(tool="search_documents", arguments=dict(query="x", limit=True)),
            dict(tool="get_document", arguments=dict(pmid="1", path="/etc/passwd")),
        ]
        for plan in plans:
            with self.assertRaises(ValueError):
                validate_plan(json.dumps(plan))


if __name__ == "__main__":
    unittest.main()
