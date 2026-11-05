# V4 사전 평가 계획

## 범위와 동결

기존 V1–V3 분류 결과를 수정하지 않는다. V4는 같은 공개 문헌의 제목·초록을 대상으로 하는 별도 검색 평가다. 기존 공식 test 문헌이 검색 저장소에 포함되지만 기존 분류 모델 학습에는 사용하지 않는다.

문헌 집합은 중복 제거한 33,645건. 출처 문헌은 SHA256(`v4-known-item-20261105` + PMID) 순서의 앞 20편이다. 검색 결과를 보기 전에 AI assistant가 원문을 읽고 lexical·paraphrase 질의를 작성했다. 앞 6편 12질의는 개발용, 뒤 14편 28영어+6한국어 질의는 평가용이다. 같은 문헌의 질의는 split을 공유한다. 독립 전문가가 판정한 gold relevance가 아니고, 알려진 출처 문헌을 다시 찾는 **known-item 탐색**이다. 다른 관련 문헌을 오답이라고 판단할 수 없으므로 precision/의학적 정답률은 주장하지 않는다.

## 검색 비교

- 원래 BM25(질문 전체 토큰 OR, 제목 가중치 3/초록 1).
- BM25(stop words 제거). 사전에 정한 영어 stop words만 사용한다.
- TF-IDF: 전체 제목+초록, word 1–2gram, sublinear TF, min_df 2, max_features 100000, L2 정규화.
- 무료 all-MiniLM-L6-v2: 모델 카드의 attention-mask mean pooling+L2, 최대 256토큰. dense cosine 전체 검색. 전문용어·한국어를 특별히 학습하지 않는다.
- hybrid: stopword BM25와 dense 상위 50개의 reciprocal rank fusion, k=60, 동등 가중치.

Hit@1/5/10, MRR@10, warm 질의 latency를 측정한다. method 선택은 **영어 개발 질의 MRR@10** 최대값, 동률 순서 BM25→hybrid→dense→TF-IDF→raw-BM25. 평가를 본 뒤 기본 method를 바꾸지 않는다. 영어/한국어 및 질의 종류별 결과를 분리한다. 문헌 그룹 단위 paired bootstrap 1000회로 hybrid−BM25 MRR@10 차이의 불확실성을 확인한다. 작은 인공 질의 표본이며 운영 검색 일반화 지표가 아니다.

## LLM 근거 검증

무료 로컬 Qwen3-4B-4bit, temperature 0, thinking off. 가중치 미세조정은 이번 V4에 포함하지 않는다. 8개 출처 기반 평가 질문 + 4개 저장소 범위 밖 질문을 실행한다. baseline 프롬프트와 구조화된 evidence-first 프롬프트, 무문맥 control을 같은 질문에서 비교한다. 각 문헌 제목+초록의 처음 1200문자, top3를 제공하고 max output 384토큰으로 고정한다. 문맥과 프롬프트 해시·실제 출력·latency를 남긴다.

검증기는 JSON schema, 제공한 PMID 여부, **실제로 모델에게 보인 텍스트에 있는 15–300문자 연속 인용**인지 검사한다. 이를 만족해도 주장에 대한 semantic entailment/의학적 정답은 보장하지 않는다. 별도 AI assistant가 실패 사례를 읽고 설명하며 사람 전문가 검토로 표시하지 않는다. 최종 응답은 검증 통과한 직접 인용을 표시하고, 생성된 주장을 검증된 사실로 승격하지 않는다. 근거가 없거나 잘못되면 유보한다.

MCP는 공식 SDK 2.x의 stdio 서버/클라이언트로 initialize→tools/list→tools/call 및 오류 경로를 실행한다. 문헌 검색·ID조회·인용 검증·통계만 노출하고 임의 SQL 실행은 노출하지 않는다. 로컬 LLM이 도구 동작/검색어를 선택하는 bounded planner를 별도로 실행하며 무제한 자율 에이전트로 부르지 않는다.
