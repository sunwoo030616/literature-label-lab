"""Post-hoc rank-fusion diagnostics; CPU recomputation, no policy retuning."""

import json
from dense_index import Encoder
from retrieval import Retriever
from settings import V4, dump


def main():
    queries = {
        q["id"]: q
        for q in json.loads((V4 / "evaluation/queries.json").read_text())["queries"]
    }
    ids = ["q09-lexical", "q08-paraphrase", "q06-paraphrase"]
    encoder = Encoder("cpu")
    vectors = encoder.encode([queries[i]["query"] for i in ids])
    retriever = Retriever()
    output = []
    for qid, vector in zip(ids, vectors):
        q = queries[qid]
        lists = {
            m: retriever.search(q["query"], m, 50, vector)
            for m in ["bm25", "dense", "hybrid"]
        }
        ranks = {
            m: next(
                (i for i, r in enumerate(rows, 1) if r["pmid"] == q["target_pmid"]),
                None,
            )
            for m, rows in lists.items()
        }
        fusion_contributions = {
            m: 1 / (60 + ranks[m]) if ranks[m] else 0 for m in ["bm25", "dense"]
        }
        target = retriever.get(q["target_pmid"])
        tokens = encoder.tokenizer.encode(target["title"] + " " + target["abstract"])
        output.append(
            dict(
                query_id=qid,
                query=q["query"],
                target_pmid=q["target_pmid"],
                top50_target_rank=ranks,
                target_rrf_contributions=fusion_contributions,
                target_token_count=len(tokens),
                dense_truncated=len(tokens) > 256,
                hybrid_top3=[
                    dict(pmid=r["pmid"], score=r["score"], title=r["title"])
                    for r in lists["hybrid"][:3]
                ],
            )
        )
    dump(
        V4 / "results/rank_fusion_diagnostics.json",
        dict(
            scope="Post-hoc CPU recomputation for explanation only; original MPS evaluation and policy unchanged",
            cases=output,
            limitations=[
                "Truncation and general-domain embeddings are possible explanations, not isolated causal tests",
                "RRF uses rank positions, not the original BM25 or cosine score magnitudes",
            ],
        ),
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    retriever.close()


if __name__ == "__main__":
    main()
