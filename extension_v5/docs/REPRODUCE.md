# 과학문헌 탐색 및 근거 검증 재현

V5는 [ETRI K2027-03 모집분야](https://etri.re.kr/preview/1793339964857/index.html)의 문헌 데이터 정제, 지식베이스 구축, 탐색 에이전트 개발을 포트폴리오에서 한 흐름으로 시연하는 독립 프로젝트다. 데이터는 [NCBI LitCovid](https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/)의 생의학 논문 제목·초록이다.

| 단계 | 구현 |
|---|---|
| 정제·검수 | Unicode NFC 및 공백 정규화, DOI 표준화, 누락 문헌 격리, 반복·상충 PMID 및 동일 본문 검수 이력 |
| 문헌 지식베이스 | 원본 CSV SHA-256·레코드 번호, PMID·DOI, 문헌 버전, 저널·출판 유형, 500자 이내 초록 구간·오프셋, 명시적 초록 소제목 |
| 탐색 | 문헌 단위 BM25 → 해당 문헌의 구간 순위화. PMID·DOI가 주어지면 그 문헌으로 제한 |
| 에이전트 | 최대 세 가지 소질문. 소질문마다 최대 세 구간을 찾고 별도 MCP 호출로 텍스트와 현재 버전을 확인 |
| 검증 | 잘못된 오프셋·출처 해시·스냅샷·인용문 거절, 모호한 DOI·없는 PMID·검색 단서 불충족을 개별 기록 |

원본 CSV와 생성 SQLite 파일은 Git에 올리지 않는다. 같은 CSV를 같은 순서로 다시 처리하면 동일한 스냅샷 ID가 나온다. 각 CSV의 `source_record`는 헤더가 1번인 **논리 레코드 번호**다. 줄바꿈을 포함한 초록은 한 레코드로 센다. 스테이징 DB의 외래 키와 두 FTS 색인을 검사한 뒤 완성 파일을 원자적으로 교체한다. 개별 구간의 위치는 정규화된 초록의 Python 문자 오프셋이다. 원문을 검증한 뒤에도 DOI 충돌이나 본문 중복·PMID 충돌을 `review_flags`에 함께 표시한다.

## 실행

저장소 루트에서 Python 3.12와 FTS5가 포함된 SQLite를 사용한다. V4 작업 환경이 있으면 재사용한다. 모델 없이 재현할 때 필요한 패키지는 `mcp==2.3.0`, `pydantic==2.14.2`다. 데이터 준비와 모델 경로는 [V4 실행 안내](../../extension_v4/docs/REPRODUCE.md)를 참고한다.

```sh
python extension_v5/src/knowledge_base.py \
  --source data/raw/train.csv data/raw/dev.csv data/raw/test.csv \
  --report extension_v5/data/ingestion.json

python extension_v5/src/science_agent.py \
  --facets extension_v5/evaluation/example.json
```

예제는 스타틴 메타분석의 PMID, 다기관 CT 연구의 DOI, 미등록 PMID를 각각 사용한다. 기대 상태는 `partial`이다. 성공한 두 소질문의 출처와 실패한 소질문의 이유가 함께 나온다. 쿼리 단어가 제목·문헌 수준에서만 일치하는 경우에는 `document_context`로 표시한다. 초록 구간의 모든 검색 단어를 요구하려면 해당 소질문에 `"require_all": true`를 설정한다.

```sh
python extension_v5/src/science_agent.py \
  --facets extension_v5/evaluation/sections_example.json
```

두 번째 예제는 같은 PMID에서 `methods`와 `results`를 따로 찾는다. 콜론이나 대문자 마침표로 표시된 실제 초록 소제목을 읽는다. 알려진 구조를 확인할 수 없는 초록은 `unstructured`로 둔다.

논문 제목을 알고 있으면 `--title`로 모델 호출 없이 정확히 일치하는 문헌을 조회할 수 있다. 동일 제목이 여러 PMID에 연결되면 후보를 반환하고 자동 선택을 보류한다.

```sh
python extension_v5/src/science_agent.py \
  --title 'Solar UV-B/A radiation is highly effective in inactivating SARS-CoV-2.'
```

로컬 Qwen의 질의 계획을 사용하려면 다음을 실행한다. Apple Silicon의 Metal 접근과 V4에서 쓰던 모델이 필요하다. `LITERATURE_LLM_PATH`로 모델 경로를 지정할 수 있다. 질문에 여러 PMID나 DOI가 들어가면 `--facets` 파일로 각 출처와 질문을 명시한다.

```sh
python extension_v5/src/science_agent.py \
  --question 'PMID 32882643에서 연구 설계와 환자 수 근거를 찾아줘.'
```

CLI의 `status`는 소질문별 **검색 및 원문 확인 상태**다. `complete`는 각 소질문에 원문을 확인한 후보가 한 개 이상 있다는 뜻이다. 그 구간이 실제 질문에 답하는지는 사람이 평가한다. 본문 PDF, 표와 그림, 시점별 최신성, 임상적 타당성은 이 초록 프로토타입의 평가 범위 밖이다.

## 평가

135,415개 구간을 초록의 오프셋과 대조했고 누락·불일치는 0건이었다. CSV 세 파일에서 각각 100개 레코드를 별도로 골라 원본과 구간 1,270개를 다시 대조했고 불일치가 없었다. [원본 출처 감사](../results/source_audit.json)

고정한 8개 개발 질문에서 지정 PMID와 기대 문자열을 모두 확인했고, 누락 PMID·모호한 DOI·필수어 불일치도 각각 구분했다. 1,200자 이후의 실제 초록에서 단어를 골라 만든 구조 점검 256건은 V5에서 전부 해당 구간을 검증했다. 이 점검의 질의는 원문에서 생성했고 V4의 앞 1,200자에 없는 단어를 고르도록 설계했다.

기존 비지정 46개 질의에서 V4 문헌 BM25는 22건, V5 구간 BM25는 20건, V5 문헌 우선 검색은 22건에서 목표 PMID를 상위 5개 결과에 포함했다. 이 표본에서는 문헌 우선 검색이 기존 발견률을 유지했고, 뒤쪽 근거를 반환할 수 있게 되었다. 초기 실패와 재검증 수치는 각각 [초기 평가](../results/evaluation_initial.json)와 [최종 평가](../results/evaluation.json)에 있다.

```sh
PYTHONPATH=extension_v4/src:extension_v5/src python -m pytest -q \
  extension_v4/tests extension_v5/tests

python extension_v5/src/audit_sources.py \
  --source data/raw/train.csv data/raw/dev.csv data/raw/test.csv \
  --output extension_v5/data/source-audit.json

python extension_v5/src/evaluate_science.py \
  --output extension_v5/data/evaluation.json

python extension_v5/src/evaluate_plans.py --sample 320 \
  --output extension_v5/data/planner-comparison.sqlite3

python extension_v5/src/summarize_plans.py \
  --checkpoint extension_v5/data/planner-comparison.sqlite3 \
  --output extension_v5/data/planner-results.json --allow-partial
```

질의 계획 실험은 기존 known-item 질의 46개와 SHA-256 순으로 고정한 320편에서 만든 영어·한국어 템플릿 질의로 구성한다. 같은 질문에서 각 프롬프트를 실행한 뒤 결과를 짝지어 비교한다. 체크포인트는 실행 중에도 저장되며 질문·프롬프트·로컬 모델 리비전을 확인하고 재개한다. 실행 시작 시의 지식베이스 스냅샷을 별도로 기록한다. 원래 지식베이스와 문헌 구성이 같은 재색인 DB로 점수를 계산할 때는 샘플과 질의가 동일한지 검사한다. 비교에서는 소질문마다 번갈아 한 건씩 선택하여 전체 **최대 세 구간**으로 고정한다. 정답은 목표 PMID 또는 템플릿에서 요구한 소질문 수이며, 전문가가 검수한 관련성이나 과학적 답변 점수는 아니다.


## 질의 계획 비교 결과

계획한 366문항 중 순차 실행을 마친 **145문항**을 두 프롬프트가 모두 있는 짝으로 비교했다. 기존 질의 46개와 고정 순서의 템플릿 99개에 해당한다. 원출력에서 확보한 검색 결과를 재색인한 동일 문헌 집합에서 확인했고, 스냅샷 ID와 결과를 기록했다. [문항별 결과](../results/planner_comparison.json)

| 비교 항목 | 기존 프롬프트 v1 | 개선 프롬프트 v2 |
|---|---:|---:|
| 전체에서 최대 3개 결과에 목표 PMID 포함 | 108/145 | 109/145 |
| PMID·DOI 미지정 질의의 목표 PMID 포함 | 59/96 | 70/96 |
| PMID 지정 질의의 해당 문헌 후보 반환 | 49/49 | 39/49 |
| 기대 소질문 수 일치 | 35/145 | 50/145 |
| 생성 시간 중앙값 | 5.61초 | 4.89초 |

PMID·DOI가 지정된 경우에는 v1을, 식별자가 없는 자연어에는 v2를 적용했다. 미지정 질의 전체에서 원문을 그대로 BM25에 넣은 대조군도 70/96이었다. 한국어로만 작성된 known-item 여섯 개에서는 원문 검색 0/6, v1 2/6, v2 3/6이었다. 제목이 주어진 별도 50개 문항의 정확 제목 조회에서는 49건이 단일 PMID로 해결됐고 한 건은 동명 후보 때문에 자동 선택을 보류했다.

실제 한국어 스타틴 질의에서는 v2가 두 소질문에서 모두 스타틴이라는 주제어를 빠뜨렸다. 3,449명 단서가 있는 소질문은 목표 PMID를 찾았지만, 연구 수 소질문은 다른 문헌을 반환했다. 따라서 원문 문자열을 확인한 후보와 질문에 대한 근거의 충분성은 구분한다. 템플릿을 포함한 소표본이며 독립 전문가가 판정한 관련성 점수는 아니다. 배치 실행 16건은 순차 실행과 검증된 출력이 같은 경우가 11건이어서 위 수치에 섞지 않았다.
