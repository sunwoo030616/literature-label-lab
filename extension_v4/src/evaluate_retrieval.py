"""Predeclared known-item ranking evaluation; lock policy using dev only."""
import json
import time
import numpy as np
from dense_index import Encoder
from retrieval import METHODS,Retriever
from settings import V4,dump,sha

def metrics(rows):
    ranks=np.array([r['target_rank'] or 0 for r in rows])
    return dict(n=len(rows),hit1=float(np.mean(ranks==1)),hit5=float(np.mean((ranks>0)&(ranks<=5))),hit10=float(np.mean((ranks>0)&(ranks<=10))),mrr10=float(np.mean(np.where(ranks>0,1/np.maximum(ranks,1),0))),latency_median_ms=float(np.median([r['milliseconds'] for r in rows])))

def main():
    lock=json.loads((V4/'results/evaluation_lock.json').read_text())
    assert all(sha(V4/p)==v for p,v in lock['sha256'].items()),'evaluation inputs changed'
    queries=json.loads((V4/'evaluation/queries.json').read_text())['queries']
    retriever=Retriever(); retriever.fit_tfidf()
    enc=Encoder(); start=time.perf_counter()
    vectors=enc.encode([r['query'] for r in queries])
    encoding_seconds=time.perf_counter()-start
    output=[]
    # Warm-up excluded; dense query encoding measured separately, not hidden in search timings.
    for method in METHODS: retriever.search(queries[0]['query'],method,10,vector=vectors[0])
    preference=['bm25','hybrid','dense','tfidf','bm25_raw']
    for split in ['dev','test']:
        if split=='test':
            dev={method:metrics([r for r in output if r['method']==method])['mrr10'] for method in METHODS}
            selected=max(preference,key=lambda method:dev[method])
            dump(V4/'results/retrieval_policy.json',dict(selected_method=selected,selection='dev English MRR@10; predefined tie order',dev_mrr10=dev,rrf_k=60,rrf_depth=50,corpus_documents=len(retriever.rows),evaluation_lock_sha256=sha(V4/'results/evaluation_lock.json')))
        for i,q in enumerate(queries):
            if q['split']!=split: continue
            # Rotate method order deterministically to reduce fixed-order cache bias.
            for method in METHODS[i%len(METHODS):]+METHODS[:i%len(METHODS)]:
                start=time.perf_counter(); found=retriever.search(q['query'],method,10,vector=vectors[i]); ms=(time.perf_counter()-start)*1000
                ids=[r['pmid'] for r in found]
                output.append(dict(query_id=q['id'],source_group=q['source_group'],split=split,kind=q['kind'],method=method,target_pmid=q['target_pmid'],target_rank=ids.index(q['target_pmid'])+1 if q['target_pmid'] in ids else None,retrieved=ids,milliseconds=ms))
        dump(V4/'results/retrieval_rows.json',output)
    summaries={}
    for split,kind in [('dev','all'),('test','all'),('test','lexical'),('test','paraphrase'),('test','korean')]:
        summaries[f'{split}-{kind}']={m:metrics([r for r in output if r['method']==m and r['split']==split and (kind=='all' or r['kind']==kind)]) for m in METHODS}
    # Correlated paraphrases/Korean grouped by source document, not independently sampled.
    groups=sorted({r['source_group'] for r in output if r['split']=='test'})
    deltas=[]
    for g in groups:
        values={m:metrics([r for r in output if r['split']=='test' and r['source_group']==g and r['method']==m])['mrr10'] for m in ['hybrid','bm25']}
        deltas.append(values['hybrid']-values['bm25'])
    rng=np.random.default_rng(20261105)
    boot=np.mean(rng.choice(deltas,size=(1000,len(groups)),replace=True),axis=1)
    dump(V4/'results/retrieval_summary.json',dict(summaries=summaries,query_encoding_seconds=encoding_seconds,encoding_queries=len(queries),latency_scope='warm search only; excludes model/index load and query encoding; dense query encoding average separately',grouped_hybrid_minus_bm25=dict(source_groups=len(groups),point_equal_group_weight=float(np.mean(deltas)),ci95=np.quantile(boot,[.025,.975]).tolist(),resamples=1000),limitations=['AI-authored source-based known-item queries; no independent relevance gold','Some non-target results can also be relevant','English encoder; Korean probes are diagnostic','TF-IDF uses full text; dense is truncated at 256 tokens']))
    print(json.dumps(summaries,indent=2)); retriever.close()

if __name__=='__main__': main()
