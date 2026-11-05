"""Reusable stdio client, used both by smoke tests and the actual LLM planner."""
import os
import sys
from contextlib import asynccontextmanager
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from settings import V4

@asynccontextmanager
async def connect_server():
    params=StdioServerParameters(command=sys.executable,args=[str(V4/'src/mcp_server.py')],env={k:v for k,v in os.environ.items() if k in {'PATH','HOME','LITERATURE_DB_PATH','TOKENIZERS_PARALLELISM'}})
    async with stdio_client(params) as (read,write):
        async with ClientSession(read,write,read_timeout_seconds=120) as session:
            initialized=await session.initialize()
            yield session,initialized

def unpack(result):
    structured=getattr(result,'structured_content',None)
    if structured is not None: return structured
    import json
    for content in result.content:
        if getattr(content,'type',None)=='text':
            try: return json.loads(content.text)
            except ValueError: pass
    return dict(error=True,content=[getattr(c,'text','') for c in result.content])
