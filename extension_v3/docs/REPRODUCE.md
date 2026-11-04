# 저장소를 읽고 재현하는 순서

## 결과 확인만 할 때

원본 V1 결과와 V2/V3 후속 결과는 각각 별도 폴더에 있습니다. 공개 원천의 문헌 ID/텍스트 해시·선택 정책·점수·원시 예측·학습 로그가 포함됩니다. 원본 데이터, 약420MB의 base encoder, 학습 checkpoint, 약2.3GB의 생성형 모델은 Git에 올리지 않았습니다. clone만으로 학습된 API가 바로 실행되는 구조는 아닙니다.

1. 루트 README와 각 버전 docs/PROTOCOL.md로 질문·후속 변경을 확인합니다.
2. locked_*policy JSON과 history로 선택 조건을 확인합니다.
3. test는 반복 사용된 개발 평가, V2 audit는 후속 내부 표본, final_internal_audit1000은 모든 모델 고정 후의 일회 내부 확인이라는 차이를 읽습니다.
4. results와 REPORT의 수치가 대응하는지 확인합니다. 모델/정책 해시는 특정 실행의 증거입니다.

## 새 환경에서 다시 실험할 때

실제 관찰 환경은 ../followup_v2/requirements-observed.txt입니다. PyTorch는 MPS가 있으면 Mac GPU, 없으면 CPU를 사용합니다. MLX 생성형 실험은 Apple Silicon이 필요합니다. Linux에서 Qwen을 실행하는 별도 backend는 구현하지 않았습니다. PDF는 한글 TrueType 폰트가 필요하며 현재 스크립트는 이 Mac의 Arial Unicode 경로를 사용합니다.

별도 clone/작업 브랜치에서 실행하세요. 재실행은 현재 clone의 results를 생성하므로 Git에 보존된 원본 결과와 구분해야 합니다. GPU/CPU backend에 따른 반올림·시간 차이까지 비트 단위로 같다고 보장하지 않습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r followup_v2/requirements-observed.txt
python src/download.py
python src/test_integrity.py
python -u src/experiment.py
python src/graph_experiment.py
```

그 후 V2 README의 diagnose → finetune → input_ablation → llm 순서를 따릅니다. Qwen은 setup_llm.py로 고정 revision을 준비하고 LITERATURE_LLM_PATH로 지정할 수 있습니다. 데이터/모델은 무료 공개 자원이며 로컬 계산 시간·디스크는 필요합니다.

verify_v2.py의 V1 preservation snapshot은 **원본 실행을 덮어쓰지 않았는지** 확인하는 검사입니다. 새로 학습해 결과가 달라진 재현 실행에서는 원본 snapshot과 일치하지 않는 것이 정상일 수 있습니다. 원본 snapshot을 조용히 갱신해 통과시키지 말고, 별도 재현 실행의 snapshot과 결과로 기록해야 합니다.

## V3 주요 실행 순서

```bash
python -u extension_v3/src/finetune512.py
python -u extension_v3/src/bag_model.py
python extension_v3/src/calibrate.py
python extension_v3/src/test_review_store.py
python extension_v3/src/test_bag_invariants.py
python extension_v3/src/test_merge.py
python extension_v3/src/smoke_service.py
python extension_v3/src/strong_tfidf.py
python -u extension_v3/src/llm_fewshot.py
python extension_v3/src/benchmark_merge.py
python extension_v3/src/chunk_sensitivity.py
python extension_v3/src/analyze_bags.py
python extension_v3/src/export_adapters.py
```

선택 모델·프롬프트가 모두 고정된 뒤 final_audit.py를 한 번 실행합니다. 결과를 보고 같은 표본에 재튜닝하지 않습니다. 원본 세션의 run_followup_pipeline.py와 run_extension_extras.py는 완료 milestone과 시간 anchor를 사용하는 운영 보조 스크립트이며, fresh clone의 단일 bootstrap 명령이 아닙니다. 보고서의 원래 추가1시간 기록을 새 실행의 시간으로 재사용하지 마세요.

## 로컬 API

프로젝트 루트에서 학습 산출물을 만든 뒤:

```bash
PYTHONPATH=extension_v3/src:followup_v2/src uvicorn service:app --host 127.0.0.1 --port 8769
```

`http://127.0.0.1:8769/docs`에서 공개 문헌 제목·초록을 등록하고, 응답 ID로 검토 라벨과 evidence 실행 이력을 확인할 수 있습니다. 검토 결과 갱신에는 현재 revision을 넣어야 합니다. smoke의 검토 라벨은 실험용 정답을 넣은 시뮬레이션이며 실제 사용자 평가로 쓰지 않습니다.
