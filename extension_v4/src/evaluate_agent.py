import asyncio
import json
from agent import run_question
from local_llm import LocalLLM
from mcp_client import connect_server
from settings import V4,dump,sha

async def main():
    tasks=json.loads((V4/'evaluation/agent_tasks.json').read_text())
    expected=json.loads((V4/'results/agent_lock.json').read_text())
    assert sha(V4/'evaluation/agent_tasks.json')==expected['tasks_sha256']
    llm=LocalLLM(); rows=[]
    async with connect_server() as (session,_):
        for task in tasks:
            result=await run_question(session,llm,task['question'])
            citations=result['user_response']['excerpts']
            target=task.get('target_pmid')
            rows.append(dict(id=task['id'],**result,target_pmid=target,target_cited=bool(target) and any(c['pmid']==target for c in citations)))
            dump(V4/'results/agent_rows.json',rows)
            print(task['id'],result.get('plan'),result.get('error'),len(citations),flush=True)
    dump(V4/'results/agent_summary.json',dict(n=len(rows),valid_plans=sum('plan' in r for r in rows),mcp_calls=sum(len(r['trace']) for r in rows),task_target_cited=sum(r['target_cited'] for r in rows),answerable_n=sum(bool(r['target_pmid']) for r in rows),abstained=sum(r['user_response']['abstained'] for r in rows),tool_errors=sum(c['is_error'] for r in rows for c in r['trace']),tool_allowlist_respected=all(c['tool'] in {'search_documents','get_document','verify_evidence'} for r in rows for c in r['trace']),scope='One planning turn and one answer turn, maximum seven read-only tool calls; diagnostic tasks, not agent generalization benchmark'))

if __name__=='__main__':asyncio.run(main())
