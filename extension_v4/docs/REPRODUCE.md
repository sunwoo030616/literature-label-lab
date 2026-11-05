# V4 실행 및 재현

저장소 루트를 작업 디렉터리로 사용한다. Python 3.11+와 FTS5 포함 SQLite가 필요하다. 이번 실행은 Python 3.12.14, Apple Silicon MPS 및 MLX에서 수행했다. 버전은 `../requirements.txt`에 고정했다. Linux/CPU에서도 검색과 MCP는 실행할 수 있으나 MLX Qwen 생성은 이 실행 코드로 지원하지 않는다.

실제 사용한 Python은 `../kkbb-ddook-ddak-back/.venv/bin/python`이다. 기존 프로젝트의 코드와 환경 패키지를 수정하지 않고 이미 설치된 패키지와 별도 V4 모델 경로를 사용했다. 새 환경에서는 별도 venv에 V4 requirements를 설치한다. 아래 `python`은 이 환경의 Python을 의미한다.

```sh
python extension_v4/src/prepare.py
# Qwen이 없다면 --qwen 추가. 약 2.3 GB 추가 다운로드.
# prepare.py --qwen 뒤 안내한 경로를 LITERATURE_LLM_PATH 환경변수에 지정한다.
python extension_v4/src/corpus.py --db extension_v4/data/corpus.sqlite3 --source data/raw/train.csv data/raw/dev.csv data/raw/test.csv --report /tmp/new-ingestion.json
python extension_v4/src/dense_index.py
python extension_v4/src/evaluate_retrieval.py
python extension_v4/src/evaluate_llm.py
python extension_v4/src/smoke_mcp.py
python extension_v4/src/evaluate_agent.py
python extension_v4/src/benchmark_sql.py
python -m unittest discover -s extension_v4/tests -v
python extension_v4/src/verify_v4.py
```

평가 스크립트는 `results`의 고정된 파일명에 기록하므로 **원본 결과를 보존할 별도 checkout**에서 전체 재실행한다. 기존 결과 확인만 하려면 마지막 verify 명령을 실행한다. 데이터/모델/생성 DB는 Git에서 제외했다. 모든 다운로드는 공개 NCBI/Hugging Face 자료이며 유료 API는 없다. 원본 CSV 해시가 달라지면 prepare는 중단한다. 모델은 revision을 지정한다.

## MCP 서버

```sh
python extension_v4/src/mcp_server.py
```

stdio transport이므로 터미널에서 일반 대화형 답변을 출력하는 프로그램이 아니다. 클라이언트 실행 명령은 `smoke_mcp.py`; 실제 initialize, tools/list, tools/call, resources/read를 별도 프로세스와 주고받는다. 도구는 `search_documents`, `get_document`, `verify_evidence`, `corpus_stats`이며 임의 SQL과 쓰기 작업은 노출하지 않는다. `LITERATURE_DB_PATH`로 DB 경로를 지정할 수 있다. 기본 retrieval method는 개발 질의로 선택해 저장한 `retrieval_policy.json`을 따른다.

## 로컬 LLM 도구 선택

```sh
python extension_v4/src/agent.py --question 'Read PMID 32425648 and find the patient count.' --output /tmp/new-question.json
```

한 번의 LLM 계획과 한 번의 근거 추출로 제한한다. read-only 도구 allowlist, 인자 검증, 최대 7회 도구 호출 제한을 적용한다. 도구/인용 오류가 발생하면 trace와 오류를 남기고 유보한다. 모델이 자유롭게 임의 코드를 실행하는 에이전트가 아니다. 원문이 포함한 지시를 무시하라는 프롬프트가 완전한 prompt-injection 방어를 보장하지 않는다.

## 프로젝트 Skill

프로젝트 `.agents/skills/literature-evidence-audit/SKILL.md`에 작업 절차와 helper를 함께 제공한다. 이 저장소를 작업 루트로 여는 Codex에서 프로젝트 Skill로 발견할 수 있는 위치다. 계정 전체 Skill이나 사용자의 MCP 앱 설정을 자동 변경하지 않았다.

```sh
python .agents/skills/literature-evidence-audit/scripts/run_audit.py
python .agents/skills/literature-evidence-audit/scripts/run_audit.py --question 'Read PMID 32425648 and find the patient count.' --output /tmp/new-skill-question.json
```

질의 helper는 출력 파일이 이미 있으면 실패하여 기존 기록을 덮어쓰지 않는다.

## 실패 뒤 추가한 옵션과 사후 진단

```sh
python extension_v4/src/analyze_quote_recovery.py
python extension_v4/src/evaluate_rewrite.py
python extension_v4/src/evaluate_agent_guard.py
python extension_v4/src/agent.py --question 'Find the study with 9 studies and 3449 patients on statins and COVID-19. How many patients were included?' --preserve-identifiers --repair-edge-ellipsis --output /tmp/new-guarded-question.json
```

`--preserve-identifiers`는 명시한 단일 PMID에 조회를 묶고, 검색어에 빠진 3자리 이상 숫자 식별자를 보존한다. 이름·작은 수·질문 의미 전체를 보존하는 보장은 아니다. `--repair-edge-ellipsis`는 앞뒤 생략부호만 제거한 문장의 유일한 정확 원문 매치를 회수한다. 중간 생략과 유사도 복구는 하지 않는다. 초기 평가 원출력을 덮어쓰지 않는 별도 옵션이다. Skill helper에도 같은 옵션을 넘길 수 있다.

사후 진단 스크립트는 이미 관찰한 질의·원출력을 재사용하므로 새 blind 평가가 아니다. 원래 프로토콜 결과와 별도 파일로 저장한다.

```sh
python extension_v4/src/evaluate_agent_guard.py --strict
python .agents/skills/literature-evidence-audit/scripts/run_audit.py --question 'Find the study with 9 studies and 3449 patients on statins and COVID-19. How many patients were included?' --require-numeric-anchors --repair-edge-ellipsis --output /tmp/new-strict-source-question.json
```

`--require-numeric-anchors`는 숫자를 검색어에 붙이는 것에 더해, 원문 후보에 그 숫자 토큰이 반드시 있도록 SQL 조건을 적용한다. 쉼표를 제거한 숫자 전체 토큰으로 비교하며, `13449`를 `3449`와 같다고 보지 않는다. lexical 검색에서만 지원한다. soft guard와 strict source 진단 결과는 각각 `agent_guard_*`, `agent_strict_*`에 보존한다.

전체 보존 해시 검사(`verify_v4.py`와 Skill의 기본 audit)는 Git에서 제외된 V1–V3 모델·배열도 로컬에 있어야 한다. 이 워크스페이스에서는 모두 확인했다. 새 clone에 이전 모델이 없으면 선행 단계의 재현이 필요하며, 파일 부재를 보존 검증 성공으로 처리하지 않는다. V4의 개별 검색·LLM·MCP 실행은 각 V4 입력만 있으면 가능하다.

포폴 PDF 재생성에는 별도로 ReportLab 5.1.0, Matplotlib 3.11.2와 이 Mac의 Arial Unicode 글꼴이 사용됐다. 본 워크스페이스에서는 `.deps`를 PYTHONPATH에 추가해 `make_portfolio.py`를 실행했다. PDF 3페이지를 PNG로 렌더링해 모두 시각 검토했다.
