# 추가 사전 계획: 특징 이웃을 반영한 관계 추정

작성 시점: 주 실험의 테스트 결과 파일 생성 전. 기존 단순 공존 관계와 비교하여, 문헌 임베딩의 이웃 구조를 반영하면 관계 보정의 효과가 달라지는지 확인한다. 주 실험 결과를 보고 고른 추가 조건이 아니다.

- T-MDML의 특징·라벨 공간 결합에서 착안했지만 bag 거리/triplet metric 학습의 재현은 아니다.
- train frozen embedding을 L2 정규화한다. 각 train 문헌에서 자기 자신을 제외한 cosine 이웃 k=[5,15,30]을 찾고 softmax(cosine) 가중 평균으로 라벨을 전파한다.
- S=softmax(y + neighbor_weighted_y), C=cosine_between_columns(S). 양의 상수가 모든 라벨 점수를 올리지 않도록 각 열의 비대각 원소 평균을 빼고 대각선을 0으로 만든다. 이 중심화는 이번 후속 실험의 선택이며 원논문 공식이 아니다.
- 같은 독립 분류기 점수에 q=clip(p+alpha*(p-prior)@C,0,1). alpha=[0,.25,.5,1,2], threshold 격자 및 tune/calibration/test는 주 실험과 동일하다.
- k와 alpha는 tune Macro-F1만으로 선택. 동일 train seed17/42/73 사용. 선택 정책 저장 후 test 평가.
- graph에 실제 이웃 대신 무작위 이웃을 연결한 1개 대조(seed20261104)도 같은 k/alpha 선택 규칙을 적용.
- 이를 'graph가 있는 것 자체'와 '의미 있는 특징 이웃'의 효과를 구분하는 탐색 실험으로 보고한다. 추가 조건 선택이 많아 검증 과적합 가능성을 명시한다.
