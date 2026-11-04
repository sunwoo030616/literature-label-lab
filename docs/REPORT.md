# 생물의학 문헌 분류: 라벨 관계의 효과와 실패 분석

실행일: 2026-11-04 (KST). 기존 개별연구의 이론 학습 이후 수행한 후속 실험.

## 결과부터
TF-IDF의 평균 Macro-F1 0.7789가 frozen BiomedBERT 0.7462보다 높았다. 라벨 관계 보정의 평균 Macro-F1 개선은 관측됐지만 안정성과 대조 조건을 함께 확인해야 한다. seed17의 paired bootstrap 95% 구간은 0을 포함해 일관된 개선의 근거가 약하다. 특징 이웃 그래프 방식의 평균 Macro-F1 차이는 +0.10pp였다.

## 데이터와 평가 설계
공식 train 24,960편 중 후보 5,000편에서 80%(4,000편)를 3회 비복원 추출했다. dev 1,000편은 tune 600 / calibration 400으로 분리했다. 공식 test 2,500편에서 초록 누락 11편·중복 1편을 제외한 2,488편을 평가했다. 테스트는 선택에 사용하지 않았다.

입력은 title+abstract만 사용했다. frozen BiomedBERT mean pooling(max 256 tokens)과 동일한 독립 Logistic Regression을 사용했다. 전면 미세조정이 아니다. TF-IDF도 학습 부분표본으로만 vocabulary를 학습했다. 스케일러·관계 행렬도 train으로만 계산했다.

## 주요 비교: 평균 ± 표본 표준편차
| 방법 | Macro-F1 | Micro-F1 | 다중주제 recall | Hamming loss |
|---|---:|---:|---:|---:|
| tfidf | 0.7789 ± 0.0082 | 0.8417 ± 0.0034 | 0.7806 ± 0.0155 | 0.0639 ± 0.0010 |
| biomedbert | 0.7462 ± 0.0139 | 0.8046 ± 0.0009 | 0.7035 ± 0.0063 | 0.0777 ± 0.0009 |
| relation | 0.7499 ± 0.0037 | 0.8209 ± 0.0046 | 0.7622 ± 0.0134 | 0.0732 ± 0.0013 |
| feature_graph | 0.7472 ± 0.0081 | 0.8117 ± 0.0064 | 0.7216 ± 0.0131 | 0.0750 ± 0.0032 |

표준편차는 3개 학습 부분표본 사이의 변동이다. 독립된 3개 테스트셋이나 공식 benchmark 5-fold 점수가 아니다.

정상 관계 vs BiomedBERT Macro-F1 평균 차이: +0.0037 (+0.37 percentage points).
무작위 라벨 대응 5개 × 학습 부분표본 3개: Macro-F1 평균 0.7464, 범위 0.7299~0.7641. 각 무작위 조건도 동일한 tune 규칙을 적용했다. 이 15개를 독립적인 통계 표본으로 가정하지 않는다.
seed17 paired bootstrap(문헌 재표집 1,000회): 차이 95% 구간 [-0.0065, +0.0107]. 다중 검정 보정이나 독립 재현은 수행하지 않았다.

## 선택한 보정 강도
alpha=0이면 관계 보정을 채택하지 않은 것이다. 개발 목표와 달라도 그대로 남겼다.
- relation, seed 17: alpha=2, Macro-F1=0.7506
- relation, seed 42: alpha=2, Macro-F1=0.7458
- relation, seed 73: alpha=2, Macro-F1=0.7532
- feature_graph, seed 17: alpha=2, Macro-F1=0.7494
- random_neighbors, seed 17: alpha=2, Macro-F1=0.7466
- feature_graph, seed 42: alpha=2, Macro-F1=0.7539
- random_neighbors, seed 42: alpha=2, Macro-F1=0.7367
- feature_graph, seed 73: alpha=0.5, Macro-F1=0.7382
- random_neighbors, seed 73: alpha=0.5, Macro-F1=0.7439

## 특징 이웃 그래프 추가 실험
train 임베딩의 cosine 이웃으로 라벨 정보를 가중 평균한 뒤 softmax 라벨 표현의 열 간 cosine 관계를 계산했다. 중심화 후 같은 확률 보정에 사용했다. k=5/15/30과 alpha는 tune에서만 선택했다. 원논문의 multi-instance/triplet 학습 재현은 아니다. 같은 구조에서 무작위 이웃을 사용하는 대조도 포함했다.
검증 선택 조건이 늘어나 과적합 가능성이 있고, 그래프 방식의 추가 효과에 대한 독립 재현은 아직 없다.
- seed 17, feature_graph: k=15, alpha=2, Macro-F1=0.7494, Micro-F1=0.8106
- seed 17, random_neighbors: k=30, alpha=2, Macro-F1=0.7466, Micro-F1=0.8102
- seed 42, feature_graph: k=5, alpha=2, Macro-F1=0.7539, Micro-F1=0.8186
- seed 42, random_neighbors: k=5, alpha=2, Macro-F1=0.7367, Micro-F1=0.8130
- seed 73, feature_graph: k=5, alpha=0.5, Macro-F1=0.7382, Micro-F1=0.8059
- seed 73, random_neighbors: k=15, alpha=0.5, Macro-F1=0.7439, Micro-F1=0.8063

## 주제별 오류: 미리 지정한 seed17
| 주제 | test support | 기본 F1 | 관계 F1 | 관계 precision | 관계 recall |
|---|---:|---:|---:|---:|---:|
| Treatment | 1029 | 0.7886 | 0.8105 | 0.8256 | 0.7959 |
| Diagnosis | 719 | 0.7629 | 0.7676 | 0.7849 | 0.7510 |
| Prevention | 924 | 0.8720 | 0.8794 | 0.8674 | 0.8918 |
| Mechanism | 563 | 0.8237 | 0.8406 | 0.8843 | 0.8011 |
| Transmission | 128 | 0.4356 | 0.4433 | 0.6000 | 0.3516 |
| Epidemic Forecasting | 41 | 0.6882 | 0.6275 | 0.5246 | 0.7805 |
| Case Report | 195 | 0.8681 | 0.8854 | 0.8995 | 0.8718 |

가능한 원인: 표본 수가 작은 라벨의 관계 추정 오차, 독립 분류기의 기존 오류 전파, 256-token 잘림, frozen 표현과 선형 분류기의 한계, train/test 주제 분포 차이. 이번 실험만으로 원인을 확정하지 않았다.

## 검토 보류: 추가 개발 기능
검증 calibration에서 문헌별 sample-F1 .85 이상·자동처리 coverage .30 이상을 목표로 임계값을 선택했다. test 성능으로 조정하지 않았다. score는 calibrated probability가 아니다.
- biomedbert: calibration 목표 달성=True, test 자동처리=73.8%, 보류 652편, 자동처리 sample-F1=0.8424, 전체 sample-F1=0.7903.
- relation: calibration 목표 달성=True, test 자동처리=93.4%, 보류 164편, 자동처리 sample-F1=0.8221, 전체 sample-F1=0.8039.

검증에서 목표를 충족했어도 테스트 retained sample-F1이 .85 미만이면 목표가 일반화되지 않은 것이다. 검토 보류의 효과와 별개로 품질 보장 실패로 기록한다. 자동처리 성능과 처리 범위는 교환관계다. 사람이 보류 문헌을 실제로 검토한 실험은 아니며 인간 검토 후 최종 정확성도 측정하지 않았다. 높은 margin도 정답을 보장하지 않는다.

## 실패 사례
아래 사례는 결과를 본 뒤 설명용으로 선정했다. 사전 고정된 예시나 인과적 설명이 아니다. 학습 정답 자체의 전문 검수도 수행하지 않았다.

### PMID 34336157
Disentangling the Association of Hydroxychloroquine Treatment with Mortality in Covid-19 Hospitalized Patients through Hierarchical Clustering.
- 정답: Treatment
- 기본: Treatment
- 관계 보정: Treatment, Diagnosis
- 과잉 예측: Diagnosis / 누락: (없음)

### PMID 34367415
Inflammation in COVID-19: from pathogenesis to treatment.
- 정답: Mechanism, Treatment
- 기본: Treatment, Mechanism
- 관계 보정: Treatment, Diagnosis, Mechanism
- 과잉 예측: Diagnosis / 누락: (없음)

### PMID 34203847
COVID-19 Vaccine Acceptance, Hesitancy, and Resistancy among University Students in France.
- 정답: Treatment
- 기본: (없음)
- 관계 보정: Prevention
- 과잉 예측: Prevention / 누락: Treatment

### PMID 34286683
Intense and Mild First Epidemic Wave of Coronavirus Disease, The Gambia.
- 정답: Prevention, Transmission
- 기본: Prevention, Transmission
- 관계 보정: Prevention, Transmission, Epidemic Forecasting
- 과잉 예측: Epidemic Forecasting / 누락: (없음)

### PMID 34201687
Occurrence of SARS-CoV-2 RNA in Six Municipal Wastewater Treatment Plants at the Early Stage of COVID-19 Pandemic in The United States.
- 정답: Prevention, Transmission
- 기본: Prevention
- 관계 보정: Prevention
- 과잉 예측: (없음) / 누락: Transmission

### PMID 34346709
Overlapping of Independent SARS-CoV-2 Nosocomial Transmissions in a Complex Outbreak.
- 정답: Transmission
- 기본: (없음)
- 관계 보정: (없음)
- 과잉 예측: (없음) / 누락: Transmission

## 관계 보정이 고친 문헌과 망가뜨린 문헌
seed17 exact-match 기준: 오답→정답 158편, 정답→오답 100편. 라벨이 달라진 문헌 472편.
유행 예측 주제는 TP 32편과 FN 9편이 그대로인데 FP가 20→29편으로 늘어 precision이 떨어졌다. 치료 주제는 FN 285→210편으로 줄었지만 FP는 114→173편으로 늘었다. 관계 보정은 누락과 과잉 예측의 교환관계를 만들었다.

## 추론 배치 실험
실행 장치: mps / macOS-26.6.2-arm64-arm-64bit. 동일한 32개 검증 문헌, batch 1/4/8, warm-up 후 각 3회. 인코더만 측정하며 네트워크 API 지연이나 동시 사용자 처리량이 아니다.
| batch | 평균 초 | documents/s | RSS MB (마지막 측정) |
|---|---:|---:|---:|
| 1 | 2.527 | 12.86 | 77 |
| 4 | 2.271 | 14.13 | 62 |
| 8 | 2.082 | 15.38 | 62 |

RSS는 CPU 프로세스의 관측 시점 메모리이며 peak GPU 메모리가 아니다. MPS allocator 메모리는 raw JSON에 별도로 기록했다. 배치 순서를 무작위화하지 않았고 background 시스템 상태를 완전히 통제하지 않았다.

## API 구현과 검증
제목/초록 입력 검증, 모델 1회 적재, 추론 동시 실행 제한(바쁠 때 429), 복수 주제·검토 권고·유사 훈련 문헌 반환. 유사 문헌은 예측 설명의 근거가 아니며 검색 품질의 정답 평가는 수행하지 않았다. API smoke 결과는 results/api_smoke.json을 확인한다.

## 선행 논문과의 경계
T-MDML은 bag 입력, 특징·라벨 공간의 graph, 전역/라벨별 metric 및 triplet 최적화를 사용한다. 이번 코드는 문헌 1벡터와 통계적 관계 후처리로 연구 질문을 축소했다. 논문 재현·성능 우위·새로운 metric learning을 주장하지 않는다. 새로 구현한 것은 대조·안정성·보류 평가 및 실행 API다.

## 재현 및 한계
실제 모델·버전·장치·가중치 해시는 results/encoding.json, 원본 해시는 data_audit.json, 문헌 ID는 split_manifest.json, 선택 정책은 locked_policies.json에 있다. 모델 및 데이터는 무료 공개 리소스를 다운로드하여 로컬에서 실행했고 유료 API를 사용하지 않았다.
학습 데이터는 공식 전체 중 제한된 부분표본이며, 테스트 중복·초록 누락을 제외했다. 단일 도메인·7라벨의 결과를 전체 과학문헌·금융문서·단백질 기능예측으로 일반화하지 않는다. calibration 표본이 작아 보류 정책 안정성은 추가 검증이 필요하다.

## 제출용 문장
개별연구에서 학습한 라벨 상관관계 개념을 생물의학 문헌 분류 실험으로 확장했습니다. 무료 BiomedBERT와 통계 분류기를 구현하고, 2,488편의 별도 테스트에서 정상·무작위 관계 및 학습 표본 변동을 비교했습니다. 성능 향상 여부와 오류를 기록하고, 검토 보류 정책과 문헌 분류·탐색 API를 구현했습니다.

이 문장은 실제 구현·실험 범위에 맞춘 초안이며, 본인이 코드·결과를 확인하고 설명할 수 있는 범위로 제출한다. 기존 이론 학습과 11월 후속 개발의 기간은 구분한다.

## 출처
- NCBI 공식 데이터: https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/
- 과제·평가 정의: https://github.com/ncbi/biocreative_litcovid
- 무료 모델: https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
- Kim, Kan, Yi. T-MDML. AAAI 2025, 17850–17857. 사용자 제공 원문.