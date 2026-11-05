"""Predeclared three-condition QA audit. Full raw outputs retained."""
import json
from settings import V4,dump,sha
from retrieval import Retriever
from evidence import validate,visible_document
from local_llm import LocalLLM


def main():
    lock=json.loads((V4/'results/evaluation_lock.json').read_text())
    assert all(sha(V4/p)==v for p,v in lock['sha256'].items())
    tasks=json.loads((V4/'evaluation/qa_tasks.json').read_text())
    policy=json.loads((V4/'results/retrieval_policy.json').read_text())
    method=policy['selected_method']
    retriever=Retriever(load_dense=method in {'dense','hybrid'})
    # Retrieve all contexts before loading the generative model. Never add target docs.
    contexts={t['id']:retriever.search(t['question'],method,3) for t in tasks}
    retriever.encoder=None
    import gc; gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
    except ImportError: pass
    llm=LocalLLM()
    output=[]
    for i,task in enumerate(tasks):
        conditions=['no-context','baseline-v1','evidence-first-v1']
        conditions=conditions[i%3:]+conditions[:i%3]
        for condition in conditions:
            rows=[] if condition=='no-context' else contexts[task['id']]
            docs={r['pmid']:visible_document(r) for r in rows}
            version='evidence-first-v1' if condition=='no-context' else condition
            payload=dict(QUESTION=task['question'],DOCUMENTS=[dict(pmid=k,text=v) for k,v in docs.items()])
            generated=llm.generate(version,payload)
            checked=validate(generated['raw_output'],docs)
            excerpts=checked['verified_excerpts']
            quoted=' '.join(r['quote'] for r in excerpts).lower()
            output.append(dict(task_id=task['id'],condition=condition,answerable=task['answerable'],target_pmid=task['target_pmid'],retrieval_method=method,retrieved_pmids=list(docs),visible_context_hashes={k:__import__('hashlib').sha256(v.encode()).hexdigest() for k,v in docs.items()},target_in_context=task['target_pmid'] in docs,**generated,**checked,target_cited=any(e['pmid']==task['target_pmid'] for e in excerpts),anchor_hits=[a for a in task['anchors'] if a.lower() in quoted]))
            dump(V4/'results/llm_rows.json',output)
            print(task['id'],condition,checked['status'],round(generated['seconds'],2),flush=True)
    summaries={}
    import numpy as np
    for condition in ['no-context','baseline-v1','evidence-first-v1']:
        rows=[r for r in output if r['condition']==condition]
        citations=[c for r in rows for c in r['citations']]
        supported=[r for r in rows if r['answerable']]
        unsupported=[r for r in rows if not r['answerable']]
        summaries[condition]=dict(n=len(rows),schema_valid=sum(r['schema_valid'] for r in rows),citation_items=len(citations),exact_valid_items=sum(c['valid'] for c in citations),supported_with_context=sum(r['target_in_context'] for r in supported),supported_target_cited=sum(r['target_cited'] for r in supported),supported_with_any_excerpt=sum(bool(r['verified_excerpts']) for r in supported),unsupported_abstained=sum(not r['verified_excerpts'] for r in unsupported),unsupported_n=len(unsupported),median_seconds=float(np.median([r['seconds'] for r in rows])))
    dump(V4/'results/llm_summary.json',dict(summaries=summaries,policy_sha256=sha(V4/'results/retrieval_policy.json'),limitations=['Verbatim provenance is not relevance or semantic entailment','AI-authored 12-task convenience audit, no clinical expert gold','No-context uses evidence-first prompt; no weights fine-tuned in V4','Abstention here refers to displayed validated excerpts; raw answers retained']))
    retriever.close()
    print(json.dumps(summaries,indent=2))

if __name__=='__main__': main()
