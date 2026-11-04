# V2 후속 진단·미세조정·생성형 LLM 사전 계획

2026-11-04. V1의 테스트 결과를 이미 확인했으므로 V2는 후속 개발 평가다. V1 보고서/결과/PDF/모델은 덮어쓰지 않는다. V1 디렉터리 파일의 실행 전 SHA256을 기록한다.

## 고정 평가 범위
- 기존 train 후보 5,000, tune600, calibration400, test2,488 manifest 유지. train seed17/42의 4,000편 부분표본은 V1과 동일.
- 새 감사 표본: 공식 train에서 V1 선택 후보와 dev/test에 없던 문헌 1,000편을 해시 순으로 고정. 중복 PMID/정규화 텍스트를 제외. 이전 모델 학습·하이퍼파라미터 선택에 사용되지 않은 내부 IID 감사 표본이다. 공식 외부 테스트와 분포가 다르므로 별도로 보고한다. 감사 라벨로 어떤 선택도 하지 않는다. 모든 모델 선택 후 평가.
- 반복 사용한 test에서는 수치 개선이 독립 확인을 의미하지 않음을 명시. 감사 표본도 동일 원천 데이터의 소규모 확인이다.

## H1 분류기 설정/과적합
- V1 frozen256 임베딩에서 raw/StandardScaler/L2 normalize × C=[.001,.01,.1,1,10] 독립 Logistic Regression. train/tune F1과 threshold 최적화 결과 저장. 선택은 tune Macro-F1만 사용.
- 비교 TF-IDF full 및 BiomedBERT tokenizer의 첫 256 token으로 제한한 TF-IDF. 후자는 단어 복원 차이가 있어 잘림의 완벽한 단독 효과는 아님.

## H2 입력 잘림
- tokenizer 길이 분포와 label별 >256/>512 비율 조사. frozen encoder mean256 vs mean512, CLS256 비교. 512와 256은 같은 표본·같은 선택 규칙 사용. 추가 길이 효과와 분류기 효과를 구분해 해석.

## H3 과제 적응 부족
- 무료 공개 BiomedBERT에서 실제 supervised multi-label BCEWithLogitsLoss 학습. mean pooling과 dropout(.1)+linear7 head.
- 조건: 마지막2개 encoder block+head 미세조정(seed17), 전체 encoder+head 미세조정(seed17), 전체 미세조정 재실행(seed42).
- 최대 3 epoch, effective batch16, max256, AdamW encoder LR2e-5/head LR1e-3, weight_decay .01, gradient clip1, 10% warmup 후 linear decay. mixed precision 없이 float32 MPS. 실패/메모리 문제 시 physical batch를 줄이고 accumulation으로 effective batch를 유지하며 기록.
- 각 epoch 끝 tune600의 Macro-F1로 checkpoint 선택. label threshold는 동일 .1~.8 격자. calibration/test/audit는 checkpoint 선택에 사용 안 함. pos_weight 미사용하여 원인 수를 제한. F1 개선을 보장하지 않음.
- partial vs full은 학습 가능한 층의 범위를 바꾸는 비교이며, full fine-tuning이 모든 frozen 문제의 원인을 단독 입증하는 것은 아님.

## H4 생성형 LLM의 실제 부가 가치
- 기존 로컬 Qwen3-4B-4bit를 무료 MLX로 실행. task model 가중치를 변경하지 않고 별도 활용.
- calibration에서 해시 순으로 선택한 고정 40편(성능/오류 여부를 보고 고르지 않음)에 주제 정의와 원문만 주어 label+verbatim evidence JSON 추출. max_input1600 token, max_output384, temperature0. schema/인용 원문일치 검사, 실패도 기록. 한 번 실패 후 추가 프롬프트 수정/재시도 없음.
- zero-shot LLM vs 동일 40편의 frozen/finetuned 분류 성능, 원문 검증 후 라벨 필터, JSON 유효율, 인용 일치율, 지연 비교. calibration 소규모 탐색이므로 외부 테스트 성능 주장 안 함. 인용 일치는 의미적으로 라벨을 지지한다는 증명 아님.
- LLM은 분류기의 예측을 보지 않고 먼저 예측하여 앵커링을 줄인다. classifier-only 결과와 별도 보고. LLM이 모델보다 낮아도 그대로 기록.

## 산출물
V1 보존 검증, 설정별 raw 결과/예측/학습 곡선/체크포인트, 최종 선택 기록, 감사 평가, 실패 사례, 설명 문서, 새 PDF. 기존 API는 변경하지 않고 미세조정 전용 API를 별도로 제공한다.

## 실행 조정 (V2 최종 결과 평가 전)
- Mac MPS의 SDPA 학습 dropout 미지원 오류를 보존하고 학습 attention을 eager로 변경했다. 모델 구조·데이터·평가 선택 규칙은 동일하다.
- pooling 비교는 512 입력의 같은 forward에서 mean/CLS를 함께 추출하여 입력 길이와 pooling을 분리한다. CLS256 대신 CLS512를 비교하며, 이를 256 pooling의 직접 검증으로 쓰지 않는다. 256 mean baseline과 512 mean을 비교하고, 512 mean과 512 CLS를 비교한다.
- 전체 층 eager 학습은 첫50 batch 출력까지 수분이 걸려 완료 시간을 크게 늘렸다. 마지막2개 층3epoch 결과를 보존한 채 전체 층의 초기 무효/미완료 실행을 중지했다. full 조건과 이후512 조건은 SDPA+attention dropout0으로 변경했다. embedding/MLP/head dropout은 유지한다. 따라서 partial(.1 attention dropout, eager) vs full(0 attention dropout, SDPA)은 순수한 학습 층 수만의 비교가 아니며, 그 차이를 결과 해석에서 명시한다. full256 vs full512는 이 설정을 맞춘다. 성능을 보고 선택한 변경이 아니라 지원 연산·실행 비용에 따른 조정이다.
- 배치2 전체 학습 프로파일: forward 약.28~.40초, backward .78~1.02초, optimizer .98~1.25초/step(warm-up 제외); GPU driver 약3.19GB. 전체 가중치 조건은 무료16GB 장비에서 반복하기에 비효율적이므로 완료 결과로 세지 않고 LoRA 조건으로 대체한다. 원본 전체 학습 중단 로그와 프로파일은 보존한다.
- 최종 적응 조건: last2 seed17(완료), LoRA seed17/42. 원본 encoder는 동결하고 모든12개 block의 query/value에 rank8, alpha16, dropout.05 adapter를 붙인다. adapter LR5e-4/head1e-3, physical/effective batch16, 최대3epoch. checkpoint/tune/calibration/test/audit 규칙은 동일. full-weight fine-tuning 성공이라고 표현하지 않는다.
- LoRA 본 실행의 첫50 batch가211.9초여서 한 epoch도 약18분이 필요한 점을 확인했다. LoRA 검증 점수를 보기 전에 무료 로컬 compute budget을 각 seed1epoch(4,000편)로 고정한다. 이미 완료한 last2는3epoch를 보존한다. LoRA256 seed17/42 및 LoRA512는1epoch, effective batch16, warmup/decay 기준은 기존750 optimizer step schedule을 유지하며250 step에서 종료한다. 완료하지 않은3epoch 전체 schedule로 표현하지 않는다. 모델 종류뿐 아니라 학습량·attention dropout도 다른 비교임을 명시한다.
