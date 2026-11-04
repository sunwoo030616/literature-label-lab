import csv
import tempfile
import unittest
from pathlib import Path
from extension_v4.src.corpus import connect, ingest, search

class CorpusTests(unittest.TestCase):
    def test_replay_dedup_quarantine_and_search(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'input.csv'
            with source.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['pmid','title','abstract','doi','label'])
                writer.writerows([
                    ['1','Vaccine study','Antibody response after vaccine.','','Prevention'],
                    ['2','Vaccine study','Antibody response after vaccine.','','Prevention'],
                    ['3','Empty','','',''],
                ])
            db = connect(Path(directory)/'corpus.sqlite3')
            first = ingest(db,source,1)
            second = ingest(db,source,1)
            self.assertEqual((first['inserted'],first['skipped'],first['quarantined']), (1,1,1))
            self.assertEqual(second['inserted'],0)
            self.assertEqual(search(db,'vaccine')[0]['pmid'],'1')
            self.assertEqual(search(db,'" OR *; DROP TABLE documents; --'),[])
            self.assertEqual(db.execute('SELECT count(*) FROM documents').fetchone()[0],1)
            self.assertEqual(db.execute('SELECT count(*) FROM documents_fts').fetchone()[0],1)
            db.close()

    def test_query_limits_and_empty_query(self):
        with tempfile.TemporaryDirectory() as directory:
            db = connect(Path(directory)/'test.sqlite3')
            self.assertEqual(search(db,''),[])
            with self.assertRaises(ValueError):
                search(db,'vaccine',51)
            db.close()

if __name__ == '__main__':
    unittest.main()
