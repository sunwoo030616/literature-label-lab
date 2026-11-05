"""One LLM tool plan -> actual MCP calls -> evidence extraction and verification.
Bounded workflow, not an open-ended autonomous agent. Only verified excerpts displayed.
"""
import argparse
import asyncio
import json
import re
import time
from evidence import validate
from local_llm import LocalLLM
from mcp_client import connect_server,unpack
from settings import V4,dump


def validate_plan(raw):
    try: plan=json.loads(raw)
    except (ValueError,TypeError): raise ValueError('planner_invalid_json')
    if not isinstance(plan,dict) or set(plan)!={'tool','arguments'} or not isinstance(plan['arguments'],dict):
        raise ValueError('planner_invalid_schema')
    tool,args=plan['tool'],plan['arguments']
    if tool=='search_documents':
        if set(args)!={'query','limit'} or not isinstance(args['query'],str) or not 1<=len(args['query'])<=1000 or type(args['limit']) is not int or not 1<=args['limit']<=3:
            raise ValueError('planner_invalid_search_arguments')
    elif tool=='get_document':
        if set(args)!={'pmid'} or not isinstance(args['pmid'],str) or not re.fullmatch(r'[0-9]{1,12}',args['pmid']):
            raise ValueError('planner_invalid_pmid')
    else: raise ValueError('planner_tool_not_allowed')
    return plan


async def run_question(session,llm,question):
    trace=[]; start=time.perf_counter()
    generated=llm.generate('planner-v1',dict(question=question))
    try: plan=validate_plan(generated['raw_output'])
    except ValueError as exc:
        return dict(question=question,planner=generated,error=str(exc),trace=trace,user_response=dict(abstained=True,excerpts=[]),seconds=time.perf_counter()-start)
    async def call(tool,args):
        if len(trace)>=7: raise RuntimeError('tool_budget_exceeded')
        result=await session.call_tool(tool,args)
        data=unpack(result)
        trace.append(dict(tool=tool,arguments=args,is_error=result.is_error,result=data))
        if result.is_error: raise ValueError('mcp_tool_error')
        return data
    try:
        result=await call(plan['tool'],plan['arguments'])
        rows=result.get('documents',[]) if plan['tool']=='search_documents' else ([result] if result.get('found') else [])
        documents={r['pmid']:r['visible_text'] for r in rows[:3]}
        answer=llm.generate('evidence-first-v1',dict(QUESTION=question,DOCUMENTS=[dict(pmid=k,text=v) for k,v in documents.items()]))
        checked=validate(answer['raw_output'],documents)
        # Explicit protocol check in addition to local context-binding validator.
        confirmed=[]
        for excerpt in checked['verified_excerpts']:
            verification=await call('verify_evidence',excerpt)
            if verification['valid']: confirmed.append(excerpt)
        return dict(question=question,planner=generated,plan=plan,answer=answer,validation=checked,trace=trace,
                    user_response=dict(excerpts=confirmed,abstained=not bool(confirmed),scope='Exact visible quotations, not semantic entailment'),seconds=time.perf_counter()-start)
    except (ValueError,RuntimeError) as exc:
        return dict(question=question,planner=generated,plan=plan,error=str(exc),trace=trace,user_response=dict(abstained=True,excerpts=[]),seconds=time.perf_counter()-start)

async def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--question',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    if len(args.question)>2000:raise ValueError('question too long')
    llm=LocalLLM()
    async with connect_server() as (session,_): result=await run_question(session,llm,args.question)
    dump(args.output,result)
    print(json.dumps(result['user_response'],ensure_ascii=False,indent=2))

if __name__=='__main__':asyncio.run(main())
