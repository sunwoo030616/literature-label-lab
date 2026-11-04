# 사전 실험 계획 — 2026-11-04

기존 개별연구의 라벨 상관관계 학습을 잇는 독립 후속 실험. T-MDML 재현이나 새로운 SOTA 모델 제안이 아니다.

질문: 고정된 생물의학 언어모델 임베딩 위에서 학습 데이터의 라벨 공존 관계로 확률을 보정하면, 독립 분류 대비 Macro-F1과 다중 주제 재현율이 개선되는가?

- 공식 NCBI BioCreative VII LitCovid train/dev/test 사용. 입력은 title+abstract만 사용. keywords, pub_type, authors, journal은 제외.
- PMID 및 정규화 title+abstract 해시로 train → dev → test 순 중복 제거. 초록 없는 문헌 제외. 제외 수 기록.
- 계산 한도: train 최대 5,000, dev 최대 1,000, test 전수. seed=20261104로 표본 고정. 공식 전체 벤치마크 점수와 직접 비교하지 않음.
- dev를 tune 60%, calibration 40%로 고정 분할. tune에서 threshold와 보정 강도만 선택. calibration은 검토 보류 정책 선택. test는 모든 선택 고정 후 실행.
- 무료 BiomedBERT (이전 PubMedBERT) frozen mean pooling, max_length=256. 전면 미세조정/생성형 LLM 아님. 임베딩 캐시.
- A: TF-IDF+독립 Logistic Regression. B: BiomedBERT+동일 분류기. C: B+train에서만 추정한 라벨 조건부 확률의 중심화 행렬 보정. D: C와 같은 보정을 하되 행렬의 라벨 대응을 무작위 순열로 섞음.
- C는 T-MDML의 feature/label graph나 metric learning을 구현한 것이 아님. 라벨 관계의 후처리 효과만 분리 검증.
- 세 번의 학습 데이터 80% 비복원 부분표본(seed 17/42/73)으로 안정성 측정. test는 같은 분할 사용. seed 17을 미리 지정한 API 모델로 사용(테스트 성능으로 선택하지 않음).
- 각 방법은 tune에서 label별 threshold를 동일 격자(0.1~0.8)로 선택. C/D alpha는 [0,.25,.5,1,2] 중 tune Macro-F1로 선택. alpha=0이 최선이면 관계 보정을 채택하지 않음.
- 지표: micro/macro-F1, label별 P/R/F1, Hamming loss, subset accuracy, AP, 다중라벨 문헌 recall, 예측 라벨 수, 오류 사례.
- 추가 개발: calibration에서 confidence boundary margin에 따른 검토 보류 정책 선택. 자동처리 문헌의 sample-F1 목표 .85, coverage 최소 .30. 달성 불가 시 명시. 보류가 정확도를 보장하지 않음. 위험은 confidence/probability calibration을 보장하지 않음.
- seed17 B/C에 paired bootstrap(test 문헌 재표집 1,000회)의 Macro-F1 차이 95% 구간. 상관관계는 인과관계가 아니며 라벨 annotation 오류도 가능.
- 서빙 실험: 동일 문헌 32개를 batch 1/4/8로 warm-up 후 3회 반복, 동일 max_length=256. 인코딩 처리량, wall time, Python RSS 및 장치 메모리(가능시). API 전체 지연/동시부하와 구분.
- 테스트를 본 후 개선한다면 새 버전 개발 결과로 분류하고 기존 결과를 덮어쓰지 않음.
