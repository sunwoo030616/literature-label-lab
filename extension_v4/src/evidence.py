"""Mechanical provenance checks. Exact occurrence does NOT establish entailment."""
import json

MAX_VISIBLE_CHARS=1200

def visible_document(row):
    return (row['title']+' '+row['abstract'])[:MAX_VISIBLE_CHARS]

def verify_quote(pmid, quote, documents):
    if not isinstance(pmid,str) or pmid not in documents:
        return dict(valid=False,reason='citation_not_in_visible_context')
    if not isinstance(quote,str) or not 15<=len(quote)<=300:
        return dict(valid=False,reason='quote_length_or_type')
    if quote not in documents[pmid]:
        return dict(valid=False,reason='quote_not_exact_visible_substring')
    return dict(valid=True,reason='exact_visible_substring_only')

def validate(raw, documents):
    try:
        obj=json.loads(raw)
    except (ValueError,TypeError):
        return dict(schema_valid=False,status='invalid_output',citations=[],verified_excerpts=[],error='invalid_json')
    if (not isinstance(obj,dict) or set(obj)!={'answer','citations'} or not isinstance(obj['answer'],str)
            or not isinstance(obj['citations'],list) or len(obj['citations'])>5):
        return dict(schema_valid=False,status='invalid_output',citations=[],verified_excerpts=[],error='invalid_schema')
    checked=[]
    for item in obj['citations']:
        if not isinstance(item,dict) or set(item)!={'pmid','quote'}:
            checked.append(dict(valid=False,reason='invalid_citation_schema'))
        else:
            checked.append(dict(**item,**verify_quote(item['pmid'],item['quote'],documents)))
    valid=[dict(pmid=i['pmid'],quote=i['quote']) for i in checked if i['valid']]
    return dict(schema_valid=True,status='grounded_excerpts' if valid else 'abstained',citations=checked,verified_excerpts=valid,
                proposed_answer=obj['answer'],proposed_answer_semantically_verified=False,
                user_response=dict(excerpts=valid,abstained=not bool(valid),scope='Verbatim visible excerpts only; relevance and entailment are not certified'))
