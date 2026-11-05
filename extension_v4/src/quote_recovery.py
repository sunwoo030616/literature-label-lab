"""Optional exact recovery of edge ellipses. No fuzzy matching or inferred text."""

from evidence import validate, verify_quote


def recover_edge_ellipsis(raw, documents):
    result = validate(raw, documents)
    recovered = []
    for item in result["citations"]:
        if item["valid"]:
            continue
        pmid, quote = item.get("pmid"), item.get("quote")
        if (
            not isinstance(pmid, str)
            or pmid not in documents
            or not isinstance(quote, str)
        ):
            continue
        candidate = quote.strip()
        changed = False
        for marker in ["...", "…"]:
            if candidate.startswith(marker):
                candidate = candidate[len(marker) :].strip()
                changed = True
            if candidate.endswith(marker):
                candidate = candidate[: -len(marker)].strip()
                changed = True
        if not changed or "..." in candidate or "…" in candidate:
            continue
        if documents[pmid].count(candidate) != 1:
            continue
        checked = verify_quote(pmid, candidate, documents)
        if checked["valid"]:
            excerpt = dict(pmid=pmid, quote=candidate)
            if excerpt not in result["verified_excerpts"]:
                recovered.append(
                    dict(
                        original_quote=quote,
                        **excerpt,
                        rule="strip_edge_ellipsis_then_unique_exact_visible_match"
                    )
                )
                result["verified_excerpts"].append(excerpt)
    result["recovered_excerpts"] = recovered
    if recovered:
        result["status"] = "grounded_excerpts"
        result["user_response"] = dict(
            excerpts=result["verified_excerpts"],
            abstained=False,
            scope="Exact visible excerpts, including explicitly recorded edge-ellipsis recovery; no entailment guarantee",
        )
    return result
