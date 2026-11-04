# 추가 개발 단계 — 긴 문헌 bag 모델과 학습 관계 규제

이 계획은 V2 완료 후 추가 개발 단계에서 실행한다. V1/V2에서 본 결과를 활용한 후속 개발이며 독립 맹검 검증으로 포장하지 않는다. 기존 결과와 파일은 보존한다. 시간 기록은 실제 clock 도구 값을 사용하고, 추가 개발 단계 시작 시 별도로 고정한다.

## 신규 연구 질문

같은 pretrained encoder에서 문헌을 chunk bag으로 바꾸고 주제별 attention을 학습하면, 한 벡터로 평균내는 방법보다 복수 주제를 잘 찾는가? 라벨 관계를 확률 후처리가 아니라 학습 regularizer에 넣으면 도움이 되는가?

## 데이터·모델

V2와 같은 train/dev/test/audit 사용. train seed17/42/73의 기존 4,000편 선택 유지. 텍스트를 tokenizer content128 이하로 나누어 최대4개 chunk로 표현한다. 4개 초과면 처음부터 끝까지 균등 간격으로4개를 선택한다. 별도의 문장 의미 분할이 아닌 token window다. 원문 전체를 항상 읽는 모델이라고 쓰지 않는다.

Frozen BiomedBERT로 chunk mean embedding 추출. A는 bag mean+튜닝 Logistic Regression, B는 label별 attention+선형 분류 BCE, C는 B에 실제 라벨 공존 양의관계로 classifier weight를 가깝게 하는 규제, D는 동일 규제에 label 대응을 섞은 관계를 사용. encoder 가중치는 고정하며 B/C/D는 작은 head만 학습한다. T-MDML의 metric/triplet을 재현한 것이 아니다.

attention head는 fixed seed, 최대30epoch, batch64, AdamW LR1e-3, dropout.1, weight_decay.01. lambda 후보 [0,.01,.1]을 tune600에서만 선택; random 대조도 같은 선택 규칙. epoch/threshold도 tune에서 선택. calibration400은 probability temperature scaling에만 사용, test/audit로 선택하지 않음.

## calibration·서비스 확장

미세조정 모델의 scalar temperature를 calibration BCE로 맞추고 test/audit의 BCE·Brier·label-level ECE를 비교. 기존 threshold 조정과 구분한다. 개선을 보장하거나 probability가 완벽하게 보정됐다고 주장하지 않는다.

검토 workflow는 문헌 content hash와 model version을 저장하고 예측·인용·검토 상태를 분리한다. 같은 요청의 중복 등록 방지, 검토 라벨 검증, 이전 revision 덮어쓰기 방지, 재시작 상태 보존을 실제 테스트한다. 신규 원문에 LLM을 호출하는 기능과 과거 LLM40편 결과를 표시하는 기능을 구분한다.

## 추가 실험

V2의 full_seed17 모델을 max_length512로 동일한3epoch 설정에서 재학습하여 fine-tuned256과 비교. 학습 비용이 늘어난 만큼 quality/latency를 함께 보고하고, 모델 선택은 tune만 사용한다. 이는 실제 추가 연구 시간에 수행하며 추가 단계가60분을 초과했는지 시작/종료 시각으로 검증한다. 불필요한 대기를 추가하지 않는다.
- 장비 프로파일에 따라 full512 조건도 full-weight 대신 LoRA512 seed17로 대체한다. query/value 모든층 rank8/alpha16, adapter LR5e-4, effective batch16; LoRA256과 설정을 맞추고 길이만 늘린다. 결과 이름도 lora512로 구분한다.
- LoRA512도 LoRA256과 같은1epoch/250 optimizer step 예산,750 step schedule 기준으로 종료한다. 추가 단계가60분 미만이면 시간을 비워 두지 않고 다른 seed의512 재현 또는 실질적 추가 검증을 수행한다.
- 512의 attention 메모리 증가를 고려해 physical batch4/gradient accumulation4로 effective16을 유지한다. 256은 physical16/accumulation1이므로 dropout RNG·연산 순서 차이도 있다. 순수한 길이 하나의 결정적 인과 효과로 해석하지 않는다.

## 최종 일회 내부 확인 표본

모든 예정 모델 및 시간 예산 기반 추가 재현을 끝내고 정책·가중치를 고정한 뒤, 공식 train에서 기존 train 후보/dev/test/audit와 PMID·텍스트 해시·DOI가 겹치지 않는1,000편을 해시 순으로 별도 선택한다. 이 최종 내부 표본 결과로 모델·프롬프트·threshold를 다시 고르지 않는다. 같은 원천 코퍼스이므로 외부 도메인 검증으로 표현하지 않는다. 모델 정책·가중치 hash를 표본 선택 전에 저장한다. 불리한 결과가 나와도 그대로 기록한다.

## 생성형 LLM few-shot 추가 대조

V2 zero-shot40 결과가 나오기 전에 추가 비교 규칙을 정한다. 동일40편을 그대로 사용하고, train seed17의4,000편 중 greedy label coverage/hash tie-break로7개 주제를 덮는 예시를 최대7개 선택한다. 예시마다220 Qwen token, 현재 문헌1600 token, 출력384 token, temperature0. 원문 인용은 현재 문헌에서만 검증한다. 프롬프트 변경/재시도 없이 raw/인용검증후 F1·JSON·인용유효율·지연 비교. 이는 Qwen 가중치 미세조정이 아닌 in-context learning이고 소규모 개발 비교다. API의 기본 프롬프트를 이40편 결과로 자동 교체하지 않는다.

## 강한 단순 baseline

V1/V2의 C1 TF-IDF를 보존하고, full/truncated256 각각 C=[.001,.01,.1,1,10]을 tune600에서만 비교한다. 같은 train seed17/4,000편과 feature 설정을 유지한다. 선택 정책을 저장한 뒤 기존test/audit를 평가하고, 최종 새 내부 확인에도 포함한다. 기존 테스트를 본 뒤 추가한 개발 비교이므로 새로운 맹검 결과로 표현하지 않는다.

## LoRA 추론 병합 검증

eval 모드에서 W_effective=W_base+(alpha/r)BA로 query/value24개 행렬을 병합한다. 학습 시 dropout을 병합하지 않고, 원본 checkpoint 파일도 수정하지 않는다. 같은32개 calibration 문헌·batch8·각 조건 warm-up·3반복으로 unmerged/merged의 처리량·확률 최대차이·문헌 라벨집합 일치율을 측정한다. 조건 순서는고정이고API전체부하가 아니므로 일반적 배속·전데이터 동일성을 보장하지 않는다. 병합 아이디어는 LoRA의 표준 배포 방식에서 가져왔다.

## 고정 모델의 문단 제거 민감도

V2가 입력 길이 진단 중인 시점에 추가 정의했다. attention_seed17 모델을 고정한 뒤 기존 audit에서 유효 chunk가2개 이상인 모든 문헌을 대상으로 높은/낮은 attention chunk 및 fixed RNG 무작위 chunk 하나를 각각 masking한다. 예측된 라벨의 attention 평균을 사용하고 무예측이면7개 라벨 평균으로 대체한다. F1·라벨 전환 비율·확률 변화를 같은 문헌에서 비교한다. attention의 인과성이나 사람이 고른 rationale의 정당성을 검증하는 실험은 아니다. 원래 문맥 삭제와 attention 재정규화가 함께 발생한다. 예시는 고정 audit 순서의 앞10편을 자동 선택하며 성공 사례를 골라내지 않는다.

## 관계 규제·attention 작동 진단

최종 선택 모델에서 라벨별 attention entropy(유효 window 수의 log로 정규화), query norm, graph penalty와 eval-mode train BCE의 비율을 기록한다. lambda0 선택 여부를 그대로 보고한다. 이는 학습 후의 서술 진단이며 작은 penalty/균등 attention만으로 성능 차이의 인과적 원인을 단정하지 않는다. 이 진단을 보고 새 lambda를 추가하지 않는다.

lambda0은 graph가 곱해지지 않아 같은 seed의 attention-only 목적함수와 정확히 같다. 동일한 학습을 반복해 시간을 쓰지 않고 seed당 한 번 학습한 zero-penalty 결과를 실제/무작위 관계 후보에서도 재사용한다. nonzero lambda의 학습과 후보 선택 규칙은 동일하다.

## 학습 방식이 맞는 uniform head 대조 추가

LoRA512가 학습 중이고 bag 특성 추출·학습·평가가 시작되기 전에 추가했다. 평균 LR 대조는 head optimizer/규제가 달라 attention의 효과만 분리하기 어렵다. 따라서 같은 AttentionHead에서 query를0으로 고정한 균등 비중 head를 추가한다. 같은 초기 classifier·BCE·dropout·AdamW·30epoch·threshold/epoch 선택 예산을 쓰며 graph penalty는0이다. 유효 chunk의 비중은 정확히1/n이다. seed17/42/73 모두 비교한다. 최종 비교는5개 조건×3학습 부분표본이며, uniform head 대비 attention의 차이와 평균 LR 참고값을 구분한다. Query가 optimizer step 후에도0이고 classifier는 갱신되는지 자동 검사한다.

## 긴 입력 실행의 메모리 대응 재시작

첫 LoRA512 실행은 batch250/1000까지 진행했지만50배치 구간 시간이69→74→92→208→320초로 악화됐다. 호스트 swap 입출력과 가용 메모리를 기록했다. 아직512의 tune/test 결과는 없었다. 이 미완료 시도는 dynamic_padding_incomplete.log로 보존하고 성공 모델로 세지 않는다. 데이터4,000편·effective16·physical4/accum4·1epoch·750step schedule/250실행은 유지한다. 재시작에서는 padding을512로 고정하고25optimizer step마다 MPS synchronize/empty_cache를 호출한다. padded 영역이 늘어 dropout RNG/연산 순서는 달라질 수 있다. 단순한 길이 하나만의 인과 비교로 해석하지 않는다. 추가 단계의 시간 anchor는 초기값을 유지한다.

근거로 참고한 PyTorch MPS 이슈: https://github.com/pytorch/pytorch/issues/201378 및 https://github.com/pytorch/pytorch/issues/181213 . 서로 다른 workload의 보고이며 이 Mac에서 같은 bug가 원인이라고 증명한 것은 아니다. torch.mps.current_allocated_memory와 driver memory는 측정 범위가 다르다: https://docs.pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory . 캐시 정리는 학습 데이터나 목적함수를 바꾸기 위한 것이 아니라 로컬 실행을 완료하기 위한 대응이다.

## 최종 내부 감사의 불확실성 — 결과를 보기 전 고정

새 최종 표본에서 fixed seed17의 다음6개 차이에 문헌 단위 paired bootstrap1,000회/percentile95% 구간을 기록한다: 튜닝 frozen−최초 frozen, last2−최초 frozen, LoRA256−튜닝 full TF-IDF, LoRA512−LoRA256, 학습 attention−uniform head, 실제 관계−무작위 관계. 최종 표본 점수로 비교 대상을 고르지 않는다. 모델/설정 선택·외부 도메인의 불확실성을 포함하지 않으며 다중 비교 보정 없는 서술적 구간이다.

bag 특성 추출에서도20배치마다 MPS synchronize/empty_cache를 호출한다. encoder.eval/inference_mode의 출력이나 학습 대상을 바꾸지 않고 오래 실행하는 로컬 encoding의 메모리 관리를 위한 조치다. driver memory는 정리 전 매 배치에서 관측한 최댓값을 별도 기록한다.
