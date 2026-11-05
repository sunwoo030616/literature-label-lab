"""Actual subprocess protocol round trips, not direct Python function calls."""
import asyncio
import json
import time
from mcp_client import connect_server,unpack
from settings import V4,dump

async def main():
    start=time.perf_counter(); checks={}; trace=[]
    async with connect_server() as (session,initialized):
        checks['initialized']=bool(initialized.server_info.name)
        listed=await session.list_tools()
        names=[t.name for t in listed.tools]
        checks['expected_tools']=set(names)=={'search_documents','get_document','verify_evidence','corpus_stats'}
        async def call(name,args):
            result=await session.call_tool(name,args)
            value=unpack(result)
            trace.append(dict(tool=name,arguments=args,is_error=result.is_error,result=value))
            return result,value
        _,stats=await call('corpus_stats',{})
        checks['real_corpus']=stats['documents']==33645
        _,found=await call('search_documents',dict(query='statin meta analysis 3449',limit=3,method='bm25'))
        checks['search_returns_documents']=bool(found['documents'])
        row=found['documents'][0]
        _,doc=await call('get_document',dict(pmid=row['pmid']))
        checks['id_lookup_matches']=doc['visible_text']==row['visible_text']
        _,verified=await call('verify_evidence',dict(pmid=row['pmid'],quote=row['visible_text'][:70]))
        checks['valid_quote_accepted']=verified['valid']
        _,bad=await call('verify_evidence',dict(pmid=row['pmid'],quote='INVENTED EVIDENCE WITH NO DOCUMENT SOURCE'))
        checks['invented_quote_rejected']=not bad['valid']
        _,missing=await call('get_document',dict(pmid='999999999999'))
        checks['missing_document_handled']=not missing['found']
        invalid,_=await call('search_documents',dict(query='vaccine',limit=999))
        checks['invalid_limit_rejected']=invalid.is_error
        injection,_=await call('get_document',dict(pmid="1' OR 1=1 --"))
        checks['invalid_pmid_rejected']=injection.is_error
        unknown=await session.call_tool('execute_sql',{'sql':'DELETE FROM documents'})
        checks['unknown_write_tool_rejected']=unknown.is_error
        resource=await session.read_resource('literature://scope')
        checks['scope_resource']=bool(resource.contents)
    report=dict(checks=checks,passed=all(checks.values()),seconds=time.perf_counter()-start,tools=names,trace=trace)
    dump(V4/'results/mcp_smoke.json',report)
    print(json.dumps(dict(checks=checks,passed=report['passed']),indent=2))
    assert report['passed']

if __name__=='__main__':asyncio.run(main())
