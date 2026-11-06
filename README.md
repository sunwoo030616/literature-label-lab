# Biomedical Literature Label Lab

개별연구에서 공부한 **bag·라벨 상관관계**를 공개 생의학 제목·초록 분류로 확장한 연구 프로젝트입니다. 최초 저성능의 원인을 조사하고 무료 부분/LoRA 미세조정, attention 대조, LLM 근거 추출과 검토 API를 실제로 구현·평가했습니다. T-MDML 전체 재현이나 SOTA를 주장하지 않습니다.

새 내부1,000편에서 최초 frozen BiomedBERT .7474→규제 조정 .7889→부분 미세조정 .8068이었습니다. 튜닝 TF-IDF도 .7930으로 경쟁력 있었고, attention·관계 규제·temperature 보정의 일관된 이득은 확인하지 못했습니다. 최종 새 내부 표본의28개 조건과 불리한 결과를 보존합니다. [먼저 읽을 결과 해설](extension_v3/docs/RESULTS_WALKTHROUGH.md)에서 범위를 확인할 수 있습니다.


## V5 과학문헌 지식베이스와 탐색 에이전트

LitCovid **PMID 33,688건**의 제목과 초록을 정제해 **125,083개 근거 구간**으로 색인했습니다. 각 구간을 원본 CSV·레코드·문헌 버전과 연결했습니다. 여러 질문을 소질문별로 검색하고 실제 MCP 도구로 인용의 출처와 위치를 다시 확인합니다. [실행·검증 안내](extension_v5/docs/REPRODUCE.md) · [평가 결과](extension_v5/results/evaluation.json)

## 개발 단계별로 보기

| 단계 | 내용 | 상태·문서 |
|---|---|---|
| V1 | 최초 baseline·라벨 관계 대조·검색 API | 완료. 아래 최초 실행 기록 및 [원본 보고서](docs/REPORT.md) |
| V2 | 저성능 원인 진단·partial/LoRA 미세조정·Qwen 인용 비교 | 완료. [보고서](followup_v2/docs/REPORT.md) / [쉬운 설명](followup_v2/docs/EXPLAIN.md) |
| V3 | 긴 문헌 LoRA·chunk bag·관계 규제·검토 이력·최종 내부 감사 | 완료. [결과 해설](extension_v3/docs/RESULTS_WALKTHROUGH.md) / [전체 보고서](extension_v3/docs/REPORT.md) / [포폴 추가본](extension_v3/output/portfolio_followup.pdf) |
| V4 | 33,645문헌 검색·무료 RAG 근거 검증·MCP·프로젝트 Skill | 완료. [쉬운 결과 해설](extension_v4/docs/RESULTS_WALKTHROUGH.md) / [재현](extension_v4/docs/REPRODUCE.md) / [포폴 추가본](extension_v4/output/pdf/rag_evidence_portfolio.pdf) |
| V5 | 문헌 데이터 검수·버전 추적·소질문별 MCP 근거 검증 | [재현 안내](extension_v5/docs/REPRODUCE.md) |

V1의 결과·보고서·모델은 해시로 보존합니다. 후속 버전은 최초 실패를 지우지 않고 별도로 비교합니다. 저장소에는 코드·정책·측정 결과를 올리고 원본 데이터·학습 가중치·SQLite·로컬 환경은 포함하지 않습니다. 생성형 Qwen의 in-context 예시와 BiomedBERT의 실제 가중치 미세조정을 구분합니다.

## V4 검색·LLM 확장 결과

동결한 질의 46개에서 5개 검색 방법을 비교했습니다. 개발 기준으로 BM25를 선택했고, 혼합 검색은 paraphrase Hit@5를 5/14→9/14로 높였지만 전체 MRR은 개선하지 못했습니다. 한국어 원질의 6개는 전부 top10 실패했으며, 같은 질의의 사후 LLM 재작성 진단에서는 6/6을 찾았습니다.

무료 로컬 Qwen의 36회 비교에서 기본/근거 우선 프롬프트의 정확 인용은 각각 6/8, 7/8이었습니다. 생략부호 복구, 숫자 누락·OR 검색의 출처 불일치, 무관하지만 정확한 인용을 별도로 분석했습니다. 실제 MCP stdio 서버/클라이언트, 제한된 LLM 도구 선택과 [프로젝트 Skill](.agents/skills/literature-evidence-audit/SKILL.md)을 제공합니다. 인용 존재와 의미적 정답을 구분하며 사후 개선을 새 blind 성능으로 표시하지 않습니다.

## V1 최초 실험 기록

## 연구 질문

같은 문헌 임베딩과 같은 분류기에서, 학습 데이터로만 추정한 주제 간 관계를 추가하면 드문 주제 및 다중 주제 문헌을 더 잘 분류하는가? 무작위 관계보다 일관되게 나은가? 자동처리를 보류하면 어떤 문헌을 얼마나 검토해야 하는가?

## 실행

Python 3.11+ 가상환경에 `pip install -r requirements.txt`. 현재 워크스페이스의 기존 `.venv`로 실행했으며 실제 버전은 `results/encoding.json`에 남깁니다. 외부 유료 API를 호출하지 않습니다. 모델·데이터 다운로드에는 인터넷이 필요합니다.

```bash
python src/download.py
python src/test_integrity.py
python -u src/experiment.py > results/run.log 2>&1
python src/graph_experiment.py
python src/analyze_errors.py
python src/make_report.py
PYTHONPATH=src uvicorn api:app --host 127.0.0.1 --port 8767
```

- `/docs`: 자동 생성 API 명세
- `POST /classify`: 제목·초록을 입력해 복수 주제, 검토 보류 여부, 관련 훈련 문헌 반환
- `GET /health`: 준비 상태
- 임베딩 추출은 로컬 Mac GPU(MPS)를 우선하고 CPU로도 실행 가능합니다.
- API의 기본 모델은 테스트 점수로 고르지 않은 seed17 모델입니다.
- 모델 가중치·데이터·임베딩·학습된 분류기는 `.gitignore`에 포함합니다. 결과·재현 코드·데이터 ID/해시는 보존합니다.

## 결과 읽기

- `docs/PROTOCOL.md`: 테스트 평가 전 고정한 실험 설계
- `results/locked_policies.json`: 테스트 평가 전에 선택·저장한 threshold/alpha/검토 정책
- `results/data_audit.json`, `split_manifest.json`: 원본 해시, 제외 수, 실제 평가 문헌 ID
- `results/test_results.json`: 3개 학습 부분표본 × 8개 비교 조건 전체 결과
- `results/paired_bootstrap.json`: seed17 관계 보정 vs 기본 모델의 문헌 단위 paired bootstrap
- `results/errors.json`: 기본 모델과 관계 보정의 테스트 오분류 비교
- `results/batch_benchmark.json`: 동등한 문헌 32개에 대한 batch 1/4/8 측정
- `docs/REPORT.md`: 해석 및 한계
- `docs/EXPLAIN.md`: 직접 설명할 수 있도록 쓴 해설
- `output/portfolio.pdf`: 기존 포트폴리오에 추가할 실험 요약

## 기존 논문과 이번 구현의 차이

원논문 T-MDML은 multi-instance multi-label 입력에서 특징·라벨 공간으로 관계를 추정하고 전역/라벨별 거리 행렬을 triplet 목적함수로 학습합니다. 이번 구현은 **문헌 한 편을 frozen encoder의 평균 벡터 하나로 표현한 multi-label 분류**입니다. Bag, triplet metric learning, 논문의 graph-based label manifold를 구현했다고 주장하지 않습니다.

학습 라벨 행렬에서 중심화된 조건부 관계 `C[i,j] = P(label_j | label_i) - P(label_j)`를 계산하고 대각선을 0으로 둡니다. 독립 분류 점수 `p`를 `clip(p + alpha * (p-prior) @ C, 0, 1)`로 보정합니다. 이는 시험할 가설을 단순하게 분리한 후처리 실험이며 새로운 알고리즘의 독창성을 주장하지 않습니다. alpha=0도 허용하며 그 경우 관계 보정을 사용하지 않습니다.

확장 개발은 **무작위 관계 대조·학습 표본 안정성·검토 보류 평가·배치 성능 측정·API 연결**입니다. 아래 점수는 확률 교정을 거치지 않았으므로 calibrated confidence나 의학적 근거로 해석하면 안 됩니다.

## 출처

- 공식 데이터: https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/
- NCBI 과제 설명 및 공식 평가 스크립트: https://github.com/ncbi/biocreative_litcovid
- 모델(MIT): https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
- 모델 설명: https://arxiv.org/abs/2007.15779
- 선행 논문: Kim, Kan, Yi. *T-MDML: Triplet-based Multiple Distance Metric Learning for Multi-Instance Multi-Label Classification with Label Correlation*. AAAI 2025, 39(17), 17850–17858. [공식 원문·DOI](https://doi.org/10.1609/aaai.v39i17.33963).

본 프로젝트는 의료 판단 도구가 아닙니다. COVID-19 문헌 주제 분류 범위의 연구 프로토타입입니다. 다른 도메인에 대한 정확성은 검증하지 않았습니다.

## 현재 워크스페이스 실행 환경

이 결과를 만든 Python은 `../kkbb-ddook-ddak-back/.venv/bin/python`입니다. 기존 깨비뚝딱 소스·환경을 수정하지 않았습니다. PDF/메모리 측정 라이브러리는 본 프로젝트의 `.deps`에 별도 설치했습니다. 일반적인 새 환경에서는 위 requirements를 설치하면 됩니다.

```bash
PYTHONPATH=.deps ../kkbb-ddook-ddak-back/.venv/bin/python src/verify_results.py
PYTHONPATH=.deps ../kkbb-ddook-ddak-back/.venv/bin/python src/smoke_api.py
```

## 특징 이웃 그래프 추가 실험

`docs/GRAPH_PROTOCOL.md`는 주 테스트 결과 생성 전에 등록한 추가 계획입니다. 원논문의 특징/라벨 공간 결합에서 착안하여 train 문헌 이웃의 라벨 전파 → 관계 추정 → 점수 보정을 별도로 비교합니다. 원논문의 triplet metric 학습과 동일하지 않습니다.

```bash
python src/graph_experiment.py > results/graph_run.log 2>&1
python src/make_report.py
```

`graph_locked_policies.json`을 저장한 뒤 `graph_test_results.json`을 생성합니다. 실제 이웃과 무작위 이웃을 각각 3개 학습 부분표본에서 비교합니다.
