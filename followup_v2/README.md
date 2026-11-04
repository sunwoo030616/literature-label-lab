# V2 — 실패 원인 진단과 무료 로컬 미세조정

V1 결과를 덮어쓰지 않는 별도 후속 개발입니다. 기존 test를 이미 본 상태라는 한계를 명시하고, 이전 학습에 쓰지 않은 내부 감사 문헌1,000편을 추가했습니다.

## 재현 실행 순서

프로젝트 루트(literature-label-lab) 기준, dependencies는 상위 requirements-lock 및 .deps를 사용합니다. 일반 환경에서는 requirements 설치 후 실행합니다. GPU 연산은 순차 실행합니다.

```bash
python followup_v2/src/diagnose.py
python followup_v2/src/feature_diagnostics.py
python -u followup_v2/src/finetune.py > followup_v2/results/finetune.log 2>&1
python -u followup_v2/src/input_ablation.py > followup_v2/results/input_ablation.log 2>&1
python -u followup_v2/src/llm_review.py > followup_v2/results/llm.log 2>&1
python followup_v2/src/verify_v2.py
python followup_v2/src/smoke_v2.py
python followup_v2/src/report_v2.py
PYTHONPATH=followup_v2/src uvicorn api_v2:app --host 127.0.0.1 --port 8768
```

LLM 실행에는 사용자가 이미 가지고 있던 `../kkbb-ddook-ddak-back/var/research_models/canvas_generator`의 무료 Qwen3-4B-4bit snapshot을 읽기 전용으로 사용합니다. 다운로드나 유료 API 호출이 아닙니다. 해당 경로가 없으면 공식 공개 모델을 준비하거나 코드의 경로를 변경해야 합니다. MLX는 Apple Silicon 전용입니다. 생성형 모델을 실행하지 않은 상태를 실행 성공처럼 표시하지 않습니다.

`src/verify_v2.py`는 V1 보존, 감사 표본 분리, 실제 encoder 가중치 변화와 결과 완전성을 확인합니다. V1 SHA256 snapshot은 최초 V2 실행 전에 이미 작성돼 있습니다. 원본 입력 데이터/임베딩 manifest를 바꾸면 보존 검사가 실패하도록 설계했습니다.

## 핵심 문서

- docs/PROTOCOL.md: 실험 계획과 런타임 조정
- docs/RESEARCH_NOTES.md: 가설·문헌 근거·확인된 사실의 구분
- docs/REPORT.md: 실제 전체 결과와 해석 (실험 종료 후 생성)
- docs/EXPLAIN.md: 본인이 이해하고 설명할 용어와 문장
- results: 각 조건의 raw 점수·예측·학습 로그·선택 정책
- models: 실제 encoder가 바뀐 checkpoint, 튜닝 분류기 (git 제외 권장)

최초 baseline은 frozen language model의 평균 벡터 위에 linear classifier를 학습한 방식입니다. 이번 fine-tuning은 마지막2개 block의 원본 가중치 또는 모든12개 block query/value의 추가 LoRA 가중치를 task loss로 업데이트합니다. 전체 원본 가중치 학습은 프로파일 후 비용 때문에 중단했으며 성공 결과로 표시하지 않습니다. 생성형 LLM의 근거 인용 실험은 별도40편의 개발 실험으로 분리합니다.

새 clone에서 생성형 실험을 재현할 때는 Apple Silicon/MLX 환경에서 다음과 같이 공개 snapshot을 별도로 준비할 수 있습니다. 이 setup 명령은 약2.3GB 다운로드를 수행하며, 이번 실행에서는 기존 snapshot을 사용해 다시 다운로드하지 않았습니다.

```bash
python followup_v2/src/setup_llm.py --destination models/qwen3-4b-4bit
export LITERATURE_LLM_PATH="$PWD/models/qwen3-4b-4bit"
python followup_v2/src/llm_review.py
```

기본 모델 경로를 환경 변수로 바꾸어도 프롬프트·문헌 선택·출력 예산은 바꾸지 않습니다. 관찰된 정확한 환경 버전은 requirements-observed.txt에 있습니다. 모델·데이터 파일은 Git에 포함하지 않으므로 clone만으로 학습된 API가 즉시 실행되지는 않습니다. 먼저 데이터/모델 준비와 학습을 순서대로 수행해야 합니다.
