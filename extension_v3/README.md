# 추가 연구·개발 단계

V2의 원인 진단·부분/LoRA 미세조정·LLM40편 비교 이후, 별도 시작 시각을 기록해 추가로 수행하는 단계입니다. 원래 결과를 덮어쓰지 않습니다.

## 실험

- `src/finetune512.py`: 모든층 query/value LoRA rank8, max512,1epoch. physical4/accumulation4로 effective16 유지.
- `src/bag_model.py`: frozen encoder의 최대4개 token window bag. mean / label attention / 실제 관계 규제 / 무작위 관계 규제를3개 학습 부분표본에서 비교. lambda0도 허용.
- `src/calibrate.py`: calibration400의 binary NLL로 temperature를 선택하고 분류 결정을 유지한 채 test/audit 확률 품질 비교.
- `src/replicate512.py`: 실제 추가 단계가60분 미만으로 끝나면 시간 기반 예산으로 다른 seed512 재현을 수행하는 선택 작업. 테스트 점수를 보고 고르는 조건이 아님.

## 실제 서비스

```bash
PYTHONPATH=extension_v3/src:followup_v2/src uvicorn service:app --host 127.0.0.1 --port 8769
```

- `POST /documents`: 실제 LoRA256 모델로 예측하고, 내용 해시+모델/보정 버전으로 등록.
- `GET /documents/{id}`: AI 예측과 사람 검토 결과를 분리 조회.
- `PATCH /documents/{id}/review`: expected_revision 기반 낙관적 동시성 제어. 알 수 없는 라벨·원문에 없는 인용 거절.
- `POST /documents/{id}/evidence`: 무료 로컬 Qwen3-4B로 주제·인용 추출. 원문과 모델 입력 안의 인용 일치를 검사하고 별도 이력 저장. 예측/사람 라벨을 덮어쓰지 않음.
- `GET /documents/{id}/evidence`: 생성형 모델 실행 이력.

Qwen은 기존 로컬 snapshot을 읽기 전용 사용합니다. LoRA classifier와 생성형 LLM은 같은 GPU lock을 공유해 동시에 실행하지 않습니다. 신뢰도 보장·의학 판단 서비스가 아닌 연구 프로토타입입니다.

## 검사·문서

`test_review_store.py`의8개 저장소 경계 검사와 `smoke_service.py`의 실제 모델/LLM 통합 검사를 수행합니다. 모델 성능, 테스트 재사용, LLM 근거의 의미적 검증 한계는 REPORT.md 및 EXPLAIN.md에서 구분합니다. 최종 결과를 먼저 고정하고 보고서를 만듭니다. `output/portfolio_followup.pdf`는 별도 포트폴리오 추가본입니다.

## 후속 비교와 재현 자료

- `src/strong_tfidf.py`: 원본 C1 결과를 보존하고 단순 모델도 tune에서 규제 강도를 선택.
- `src/llm_fewshot.py`: 같은40편에서 train-only 예시를 넣은 in-context 대조. Qwen 미세조정은 아님.
- `src/benchmark_merge.py`: LoRA 병합 전후 실제 확률·라벨·처리량 측정. 원본 파일 보존.
- `src/chunk_sensitivity.py`: 고정 attention 모델에서 높은/낮은/무작위 문단 제거 비교. 인과적 근거 검증은 아님.
- `src/export_adapters.py`: base model이 별도로 필요한 adapter+head safetensors export와 tensor round-trip 확인.
- `src/final_audit.py`: 모든 정책을 고정한 뒤 겹치지 않는 새 내부1,000편을 일회 확인.
- 실행 환경 버전은 `../followup_v2/requirements-observed.txt`에 보존. GPU 작업은 순차 실행.
