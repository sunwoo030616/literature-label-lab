"""Human-readable V4 results derived from saved runs, with failed cases retained."""

import json
from settings import V4


def read(name):
    return json.loads((V4 / "results" / name).read_text())


def main():
    r = read("retrieval_summary.json")
    s = r["summaries"]
    l = read("llm_summary.json")["summaries"]
    a = read("agent_summary.json")
    w = read("rewrite_summary.json")
    q = read("quote_recovery.json")
    sql = read("sql_benchmark.json")
    lines = [
        "# V4 결과 해설: 검색·근거 검증·MCP·Skill",
        "",
        "이번 확장은 기존 문헌 분류 모델을 그대로 두고, 필요한 문헌을 찾고 인용을 확인하는 별도 시스템을 만든 것이다. 무료 공개 MiniLM과 로컬 Qwen3-4B-4bit를 사용했고 유료 API는 호출하지 않았다. V4에서는 모델 가중치를 추가 미세조정하지 않았다.",
        "",
        "## 1. 무엇이 실제로 실행됐나",
        "",
        "- 실제 LitCovid CSV 33,699행을 읽어 고유 문헌 33,645건을 SQLite/FTS5에 저장했다. 중복 43행과 빈 초록 11행을 분리했고, 동일 파일 재실행의 추가 문헌은 0건이었다.",
        "- 무료 MiniLM으로 전체 문헌 33,645건을 384차원 벡터로 인코딩했다. attention-mask mean pooling, L2 정규화, 256토큰 제한을 사용했다.",
        "- 검색 결과를 보기 전에 출처 문헌 20편의 질의 46개, LLM 질문 12개와 프롬프트를 고정해 커밋했다. 문헌별로 개발 6편/평가 14편을 나눴다.",
        "- 검색 5방법 × 46질의 = 230개 순위 기록, LLM 3조건 × 12문항 = 36개 원출력, 실제 MCP 에이전트 진단 6개를 보존했다.",
        "",
        "## 2. 검색: 복잡한 방법이 항상 이기지 않았다",
        "",
        "| 방법 | 개발 12질의 MRR@10 | 평가 34질의 MRR@10 | 평가 Hit@5 |",
        "|---|---:|---:|---:|",
    ]
    for m in ["bm25_raw", "bm25", "tfidf", "dense", "hybrid"]:
        lines.append(
            f"| {m} | {s['dev-all'][m]['mrr10']:.3f} | {s['test-all'][m]['mrr10']:.3f} | {s['test-all'][m]['hit5']:.3f} |"
        )
    lines += [
        "",
        "MRR@10은 지정한 출처가 1등이면 1, 2등이면 1/2, 10위 밖이면 0으로 채점한 뒤 평균한 값이다. Hit@5는 그 출처가 상위 5개 안에 있었는지 센 것이다. 개발 질의 MRR@10과 사전 동률 규칙으로 **BM25**를 기본 검색으로 선택했다. 평가에서 raw BM25가 약간 높았지만 기본 method를 바꾸지 않았다.",
        "",
        "| 질의 종류 | 수 | BM25 Hit@5 | MiniLM Hit@5 | Hybrid Hit@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in ["lexical", "paraphrase", "korean"]:
        v = s["test-" + kind]
        lines.append(
            f"| {kind} | {v['bm25']['n']} | {v['bm25']['hit5']:.3f} | {v['dense']['hit5']:.3f} | {v['hybrid']['hit5']:.3f} |"
        )
    ci = r["grouped_hybrid_minus_bm25"]
    lines += [
        "",
        "**왜 이런 결과가 나왔나?** 질의에 전문용어·숫자·고유명이 그대로 있으면 BM25가 강했다. 의미 검색 모델은 영어 일반 문장용이고 긴 초록은 처음 256토큰으로 제한돼 전문 분야·문서 길이에서 불리할 수 있다. 이 두 요인의 인과적 기여를 분리한 실험은 아니므로 단일 원인으로 단정하지 않는다.",
        "",
        "혼합 검색은 점수 크기가 아닌 순위를 섞는 RRF(k=60, 각 top50, 같은 가중치)다. 강한 lexical 1위도 dense에서 밀리면, 두 목록에 중간 순위로 동시에 등장한 다른 문헌보다 낮아질 수 있다. 실제 q09-lexical은 BM25 1위에서 hybrid top10 밖으로 밀렸고, q08-paraphrase는 두 단일 방법의 top10 밖에서 hybrid 1위로 들어왔다. 상세한 top50 순위와 기여도는 `rank_fusion_diagnostics.json`에 있다.",
        "",
        f"문헌 그룹별 같은 비중으로 계산한 hybrid−BM25 MRR@10 차이는 {ci['point_equal_group_weight']:.3f}, 그룹 bootstrap 95% 구간은 [{ci['ci95'][0]:.3f}, {ci['ci95'][1]:.3f}]로 0을 포함한다. 전체 질의 평균 차이와 가중 방식이 다르다. 전반적인 개선을 주장할 근거는 부족하다.",
        "",
        "**이 평가는 전문가 relevance gold가 아니다.** AI assistant가 출처를 읽고 작성한 known-item 질의다. 해당 문헌 하나를 다시 찾는 실험이며 다른 관련 논문이 검색돼도 오답이라고 단정할 수 없다. 사용자 검색 로그나 독립 판정자 데이터도 없다. 영어 질문의 큰 어휘 중복 때문에 lexical 방법에 유리할 수 있다.",
        "",
        "## 3. 무료 LLM: 답변과 출처 검사를 분리했다",
        "",
        "문헌이 없는 조건, 기본 RAG 프롬프트, 근거 우선 프롬프트를 비교했다. top3 문헌마다 제목+초록의 처음 1200문자만 제공했다. JSON 구조, 제공한 PMID, 실제 보인 텍스트의 15–300문자 연속 인용인지 검사했다. 프롬프트 버전·해시·모델 revision·원출력·시간을 남겼다.",
        "",
        "| 조건 (각 12회) | JSON schema | 정확 인용 / 생성 인용 | 지정 출처 인용 / 8 | 범위 밖 유보 / 4 | 중앙 생성 시간(초) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, value in l.items():
        lines.append(
            f"| {condition} | {value['schema_valid']}/12 | {value['exact_valid_items']}/{value['citation_items']} | {value['supported_target_cited']}/8 | {value['unsupported_abstained']}/4 | {value['median_seconds']:.2f} |"
        )
    lines += [
        "",
        "두 RAG 조건 모두 지정 출처가 문맥에 들어간 질문은 7/8개였다. 근거 우선 프롬프트에서 엄격한 인용이 1개 더 통과했지만 작은 표본이다. 프롬프트를 바꾸면 항상 좋아진다고 일반화하지 않는다. 무문맥 조건은 질문에 답할 배경 지식이 있어도 공급 문헌이 없으므로 모두 인용을 유보했다. 유보는 최종 표시할 검증 인용이 없다는 뜻이며, raw answer는 별도로 보존했다.",
        "",
        "### 실패 사례 A: 정답처럼 보이지만 인용이 바뀜",
        "",
        "qa06의 태양 UV 질문에서 두 프롬프트 모두 답을 `less than 2 min`으로 썼다. 그러나 인용을 `...in less than 2 min.`처럼 만들어 앞의 생략부호가 원문에 없었다. baseline의 qa08에도 같은 형태가 발생했다. 엄격한 검사는 모두 거절했다.",
        "",
        f"후속 구현은 앞뒤 생략부호만 제거하고 **남은 전체 문장이 보이는 원문에서 정확히 한 번 발견될 때만** 복구하는 옵션이다. 중간 생략부호, 오탈자 유사도, 중복 매치는 허용하지 않는다. 원래 36회 결과는 변경하지 않고 {q['new_exact_excerpts']}개 인용을 별도 사후 분석에서 복구했다. 복구 뒤 지정 출처 인용은 두 RAG 조건 모두 7/8이 됐다. 새 blind 질문으로 재평가한 개선은 아니며, 의미적 타당성을 인증하지도 않는다.",
        "",
        "### 실패 사례 B: 평가 질문이 출처를 충분히 특정하지 못함",
        "",
        "qa01은 스타틴 메타분석의 연구 수·환자 수를 물었으나 어떤 논문인지 특정하지 않았다. 모델은 지정 출처(PMID 32882643, 9연구/3449명)가 아닌 실제 검색된 다른 메타분석(PMID 32975814, 10연구/10233명)을 정확히 인용했다. 이것을 곧바로 환각이나 사실 오답으로 분류하면 평가가 과장된다. 표에서는 지정 출처를 인용하지 못한 것으로만 센다. 소스가 중요한 질문은 PMID·제목 등 식별자를 명시해야 한다.",
        "",
        "**최종 사용자가 받는 결과는 검증된 직접 인용과 PMID다.** 생성한 서술은 proposed_answer로 분리하고 semantically_verified=false를 유지한다. 실제 인용이라도 질문에 무관하거나 문헌 자체의 주장에 오류가 있을 수 있다.",
        "",
        "## 4. 한국어 실패 → 검색어 재작성 진단",
        "",
        f"영어 저장소에서 한국어 원질의 6개는 모든 초기 방법의 top10에서 실패했다. 동결한 planner-v1으로 영어 검색어를 만들고 MCP BM25 검색을 호출하자 top10 출처 탐색은 0/6 → {w['rewritten_hit10']}/6, top5는 {w['rewritten_hit5']}/6이었다. 검색어 생성 평균 {w['mean_generation_seconds']:.2f}초가 추가됐다.",
        "",
        "이것은 이미 실패를 본 같은 6질의를 다시 사용한 **사후 진단**이다. 한국어 검색의 새로운 blind 성능이 아니다. 질문을 번역·재작성하는 비용과 정보 손실 가능성도 생긴다. 기존 검색 정책과 질의를 변경하지 않았다.",
        "",
        "## 5. 실제 MCP·Skill과 에이전트 범위",
        "",
        "- 공식 Python MCP SDK 2.3.0으로 stdio 서버/클라이언트를 구현했다. HTTP처럼 포장한 임의 함수 호출이 아니라 별도 프로세스 initialize, tools/list, tools/call, resources/read를 실제 실행했다.",
        "- 도구: `search_documents`, `get_document`, `verify_evidence`, `corpus_stats`. 파라미터 타입·길이·범위를 검사하고 DB는 read-only 연결한다. 임의 SQL 실행이나 쓰기 도구는 없다.",
        "- 프로토콜 smoke: 검색/조회/인용/누락 문헌/잘못된 인자/존재하지 않는 쓰기 도구/scope resource 등 12항목 통과.",
        f"- 로컬 LLM 도구 선택 진단: 유효 계획 {a['valid_plans']}/6, 실제 MCP 호출 {a['mcp_calls']}회, 지정 출처 인용 {a['task_target_cited']}/{a['answerable_n']}, allowlist 준수 {a['tool_allowlist_respected']}. 한 번의 계획과 한 번의 근거 추출, 최대 7회 read-only 도구 호출로 제한했다.",
        "- 프로젝트 Skill은 `.agents/skills/literature-evidence-audit`에 SKILL.md, 실행 helper, UI 메타데이터로 제공한다. 기본 선택은 허용한다. 계정 전체 설치나 앱 MCP 설정 변경은 하지 않았다.",
        "",
        "도구 선택 실패도 trace에 남는다. 작은 6개 진단은 agent benchmark가 아니고, prompt injection 한 사례를 통과했다고 완전한 방어를 입증하지 않는다. 논문 전문 QA나 KG-RAG, 다중 에이전트 시스템이라고 주장하지 않는다.",
        "",
        "## 6. DB 쿼리를 어떻게 설명하나",
        "",
        f"같은 PMID 40개를 동일 결과로 번갈아 조회했다. `WHERE pmid=?` 인덱스 조회 중앙값은 {sql['milliseconds']['indexed_pmid']['median']:.4f}ms, `WHERE CAST(pmid AS TEXT)=?` 대조 조회는 {sql['milliseconds']['expression_pmid']['median']:.2f}ms였다. EXPLAIN에서 각각 SEARCH/SCAN을 확인했다. 일부러 인덱스를 못 쓰게 한 대조 조건이며, 기존 느린 운영 쿼리를 고쳐 얻은 성과나 일반적인 배수 개선으로 쓰면 안 된다.",
        "",
        f"비어 있지 않은 DOI가 중복된 그룹 {sql['duplicate_nonempty_doi_groups']}개도 확인했다. PMID/내용 중복과 DOI 중복을 같은 의미로 처리하지 않고 기록했다. 수정 문헌 이력·외부 증분 수집·분산 처리·백만 건 부하 검증은 후속 과제다.",
        "",
        "## 7. 면접에서 1분 설명",
        "",
        "> 개별연구에서 문헌의 bag과 라벨 관계를 공부한 뒤, 공개 문헌 분류의 실패 원인을 실험하고 문서 검색과 근거 검증으로 확장했습니다. 3만여 문헌의 재실행 가능한 적재 파이프라인과 5가지 검색 대조를 만들었습니다. 혼합 검색이 키워드 질문의 순위를 떨어뜨리는 경우와 한국어 실패를 발견했고, 무료 LLM 검색어 재작성과 정확 인용 복구를 별도 사후 진단으로 검증했습니다. 프롬프트 버전, 실제 MCP 도구 호출, 재현 Skill과 원출력을 함께 남겨 결과가 좋아 보이는 것보다 어떤 조건에서 신뢰할 수 있는지 설명하는 프로젝트로 발전시켰습니다.",
        "",
        "AI 도구를 활용한 구현·분석 과정과 본인이 실제 실행하고 이해한 범위를 사실대로 설명한다. 과거 개별연구 때 이 전체 시스템을 이미 구현했다고 소급해서 말하지 않는다.",
        "",
        "## 출처와 재현",
        "",
        "- [공식 LitCovid CSV](https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/)",
        "- [MiniLM 공식 모델 카드](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): mask mean pooling·정규화 및 영어 모델 범위",
        "- [MCP 공식 Python SDK v2](https://py.sdk.modelcontextprotocol.io/): 실제 설치 버전의 MCPServer 인터페이스 사용",
        "- [SQLite FTS5 공식 문서](https://www.sqlite.org/fts5.html): BM25·외부 content 인덱스",
        "- `REPRODUCE.md`: 실행 환경·무료 모델 준비·검증 명령",
        "- `../results/evaluation_lock.json`: 질의·프롬프트 해시",
        "- `../results/llm_rows.json`, `agent_rows.json`: 실제 원출력과 trace",
        "- `../results/quote_recovery.json`: 사후 복구 결과. 기존 LLM 기록은 보존",
        "- `../results/verification.json`: 원출력 재검증과 이전 버전 해시 확인",
        "",
    ]
    guard = read("agent_guard_summary.json")
    strict = read("agent_strict_summary.json")
    position = lines.index("## 5. 실제 MCP·Skill과 에이전트 범위")
    lines[position:position] = [
        "### 추가 실패: 숫자를 붙여도 OR 검색은 출처를 강제하지 않는다",
        "",
        "원래 English agent는 질문의 3449명을 검색어에서 빠뜨렸다. 숫자를 덧붙이는 soft guard를 만들었지만 OR 검색에서는 숫자가 없는 논문도 상위에 남았다. 지정 출처 인용은 3/4로 그대로였다. 더 심각하게 모델은 질문에서 3449라는 답을 가져오면서 실제 문맥의 “hydrogen bonds” 문장을 인용했다. 이 인용은 원문에 실제 있어 occurrence 검사에는 통과했지만 환자 수를 뒷받침하지 않는다. AI assistant가 raw trace를 읽어 진단했으며 독립 전문가 판정은 아니다. 생성 답변을 미검증으로 분리한 이유다.",
        "",
        f"후속 strict 옵션은 질문에서 추출한 3자리 이상 숫자를 원문 후보가 반드시 포함하게 하는 parameterized SQL 조건이다. gold PMID를 넣지 않고, 동일한 원래 planner 출력에 적용했다. 원형 {strict['original_target_cited']}/4 → soft guard {guard['guarded_target_cited']}/4 → strict source {strict['guarded_target_cited']}/4의 지정 출처 인용으로 바뀌었다. 변경된 계획만 다시 검색·생성하고 나머지 답변은 재사용했다. 새 blind 평가가 아니며 숫자 표기가 다르거나 숫자가 검색 의도와 무관하면 오히려 누락될 수 있다. 의미 검증을 대체하지 않는다.",
        "",
    ]
    (V4 / "docs/RESULTS_WALKTHROUGH.md").write_text("\n".join(lines))
    print(V4 / "docs/RESULTS_WALKTHROUGH.md")


if __name__ == "__main__":
    main()
