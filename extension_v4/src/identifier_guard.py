"""Optional deterministic source-binding guard, separate from the frozen baseline."""

import re
from copy import deepcopy


def preserve_identifiers(plan, question, require_numeric=False):
    effective = deepcopy(plan)
    pmids = re.findall(r"\bPMID\s*[:#]?\s*(\d{1,12})\b", question, re.IGNORECASE)
    if len(set(pmids)) > 1:
        raise ValueError("multiple_explicit_pmids_require_disambiguation")
    changes = []
    if pmids:
        bound = dict(tool="get_document", arguments=dict(pmid=pmids[0]))
        if effective != bound:
            changes.append(dict(rule="bind_explicit_pmid", value=pmids[0]))
        return bound, changes
    if effective["tool"] == "get_document":
        raise ValueError("unrequested_pmid_lookup")
    # Keep 3+-digit source anchors (patient counts, years), including Korean suffixes.
    # Small counts, drug names and semantic constraints are NOT comprehensively checked.
    anchors = re.findall(r"(?<![A-Za-z0-9_-])\d[\d,]{2,}(?![A-Za-z0-9_-])", question)
    for anchor in dict.fromkeys(anchors):
        if anchor.replace(",", "") not in effective["arguments"]["query"].replace(
            ",", ""
        ):
            effective["arguments"]["query"] += " " + anchor
            changes.append(dict(rule="preserve_numeric_anchor", value=anchor))
    if len(effective["arguments"]["query"]) > 1000:
        raise ValueError("guarded_query_too_long")
    if require_numeric and anchors:
        effective["arguments"]["required_numeric_anchors"] = list(
            dict.fromkeys(anchors)
        )
        changes.append(
            dict(
                rule="require_numeric_source_match", values=list(dict.fromkeys(anchors))
            )
        )
    return effective, changes
