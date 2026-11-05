"""Matched read-only SQL benchmark: indexed lookup vs non-sargable expression."""
import json
import sqlite3
import time
import numpy as np
from settings import DB,V4,dump


def main():
    db=sqlite3.connect(DB.resolve().as_uri()+'?mode=ro',uri=True)
    queries={
        'indexed_pmid':'SELECT title FROM documents WHERE pmid=?',
        'expression_pmid':'SELECT title FROM documents WHERE CAST(pmid AS TEXT)=?',
    }
    ids=[r[0] for r in db.execute('SELECT pmid FROM documents ORDER BY pmid LIMIT 40 OFFSET 1000')]
    timings={name:[] for name in queries}; plans={}
    for name,sql in queries.items():
        plans[name]=[list(r) for r in db.execute('EXPLAIN QUERY PLAN '+sql,(ids[0],))]
        db.execute(sql,(ids[0],)).fetchall()
    for i,pmid in enumerate(ids):
        values={}
        names=list(queries) if i%2==0 else list(reversed(queries))
        for name in names:
            start=time.perf_counter();values[name]=db.execute(queries[name],(pmid,)).fetchall()
            timings[name].append((time.perf_counter()-start)*1000)
        assert values['indexed_pmid']==values['expression_pmid']
    analytics={}
    for label in ['Treatment','Diagnosis','Prevention','Mechanism','Transmission','Epidemic Forecasting','Case Report']:
        analytics[label]=db.execute("SELECT count(*) FROM documents WHERE ';'||labels||';' LIKE ?",('%;'+label+';%',)).fetchone()[0]
    duplicate_doi=db.execute("SELECT COUNT(*) FROM (SELECT doi FROM documents WHERE doi<>'' GROUP BY doi HAVING COUNT(*)>1)").fetchone()[0]
    report=dict(rows=db.execute('SELECT count(*) FROM documents').fetchone()[0],repeats=len(ids),queries=queries,plans=plans,milliseconds={k:dict(median=float(np.median(v)),p95=float(np.quantile(v,.95)),raw=v) for k,v in timings.items()},label_supports=analytics,duplicate_nonempty_doi_groups=duplicate_doi,limitations=['Warm local SQLite read-only benchmark, alternating order; not distributed/production scale','Expression predicate is a deliberately non-indexable same-result control','Ingestion deduplicates PMID and content; DOI conflicts remain visible and are not silently merged'])
    dump(V4/'results/sql_benchmark.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='milliseconds'},indent=2))
    print({k:round(v['median'],4) for k,v in report['milliseconds'].items()})
    db.close()

if __name__=='__main__':main()
