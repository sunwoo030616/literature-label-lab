# 원논문에서 이번 구현으로 이어진 부분

사용자가 제공한 T-MDML 원문을 재확인했습니다. 아래 페이지는 논문 인쇄 쪽수 기준입니다.

| 원문 요소 | 원논문의 역할 | 이번 후속 구현 | 동일하게 구현하지 않은 부분 |
|---|---|---|---|
| bag·bag-level 복수 라벨,17850~17852 | instance마다 정답을 요구하지 않고 bag 전체에 라벨 부여 | 문헌의 token window 여러 개를 bag으로 표현하고 문헌 전체 주제만 학습 | 원논문의10개 benchmark·원본 instance 특성은 사용하지 않음 |
| 특징/라벨 기반 관계 추정, 식2~6,17852 | bag 이웃의 라벨을 평활화하고 label manifold의 column cosine으로 관계 추정 | V1 별도 feature-neighbor 후처리 실험. V3는 train label co-occurrence cosine을 사용 | V3 관계를 원논문의 feature-label joint estimator라고 부르지 않음 |
| 전역·라벨별 transformation, 식7~10,17853 | target/imposter bag triplet으로 L0+Ll을 학습 | 주제별 attention query와 classifier vector를 BCE로 학습 | triplet·Mahalanobis transformation·NN 분류를 구현하지 않음 |
| 관계 규제, 식11,17853 | 관련 라벨의 transformation matrix가 가까워지도록 제약 | 관련 라벨 classifier vector의 제곱 차이를 가중 penalty로 추가 | 학습 대상은 matrix Ll이 아닌 classifier vector이며 동일 알고리즘이 아님 |

## 후속 구현에서 실제로 바꾼 질문

원논문은 “어떤 거리에서 bag끼리 유사한가?”를 학습합니다. 이번 후속 모델은 “한 문헌의 어떤 조각을 주제별로 다르게 합치면 분류에 도움이 되는가?”를 시험합니다. 관계 penalty가 도움이 되는지 실제 관계·무작위 관계·관계 없음으로 비교하고, attention 자체의 효과는 같은 학습 방식의 uniform head로 확인합니다.

V3 관계 penalty는 `sum(G[i,j] * mean_d((w_i-w_j)^2)) / sum(G)`입니다. 분류 loss의 단위에 대한 상대 크기도 별도 분석합니다. 관련된 라벨의 w를 가깝게 하는 유도는 분리해야 할 주제까지 비슷하게 만들 수 있으므로 향상을 전제하지 않습니다.

## 프로젝트의 세 갈래는 구분합니다

1. frozen/partial/LoRA 단일 문헌 분류: encoder 과제 적응과 입력 길이를 비교합니다.
2. frozen encoder의 문헌 bag 모델: head의 pooling·attention·관계 규제를 비교합니다. LoRA와 bag을 end-to-end로 결합해 학습한 것은 아닙니다.
3. 생성형 Qwen 근거 추출: 주제와 원문 인용을 별도로 생성합니다. 서비스는 미리 지정한 LoRA256 분류 결과와 Qwen 이력을 구분해 보관합니다.

이 세 가지 점수를 합쳐 하나의 모델 성능처럼 쓰지 않습니다. 실제 서비스 기본값도 최종 감사에서 점수가 가장 높은 조건으로 자동 교체하지 않습니다.
