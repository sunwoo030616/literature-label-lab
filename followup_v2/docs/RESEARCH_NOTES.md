# 후속 원인 진단을 위한 문헌·구현 검토

## 이 프로젝트에서 직접 확인한 사실

- 동일 frozen256 임베딩에서 standard scaling+C1은 train Macro-F1 .981, tune .776이었다. C=.01은 train .869, tune .813이었다. 규제를 강하게 해서 train/tune 간격이 줄었다.
- tune으로 고른 C=.01 설정의 재사용 테스트 점수는 .793이다. V1 seed17 .748과 같은 표본 비교다. V1 3seed 평균 .746과 혼동하지 않는다.
- tokenizer로 256 token까지 제한한 TF-IDF가 full TF-IDF보다 이번 테스트에서 높았다. 'BERT가 문장을 잘라서 낮았다'는 한 가지 설명만으로는 맞지 않는다. 잘림과 vocabulary 변화가 함께 있으므로 원인을 단정하지 않는다.

## 외부 문헌에서 얻은 가설과 경계

1. Microsoft 공식 모델 카드는 BiomedBERT를 PubMed 초록으로 사전학습한 masked language model로 설명한다. 이번 V1은 그 가중치를 고정하고 평균 임베딩에 선형 분류기를 얹었으므로, 공식 논문의 fine-tuned biomedical benchmark 성능을 곧바로 기대할 수 없다.
   https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
2. Ditto(EMNLP2023)는 frozen BERT의 문장 표현에 비정보성 단어 편향이 있을 수 있음을 연구한다. 반면 EMNLP2022의 분석은 anisotropy 자체를 근본 원인으로 단정하는 해석에 반대한다. 따라서 우리 실험에서 cosine 분포만 보고 'anisotropy가 원인'이라고 쓰지 않는다. 직접 검증하는 것은 pooling·정규화·규제·과제 적응의 영향이다.
   https://aclanthology.org/2023.emnlp-main.359/
   https://aclanthology.org/2022.findings-emnlp.314/
3. Attention-based MIL(ICML2018)은 bag의 instance를 attention으로 합치는 방법을 연구한다. 후속으로 문헌을 여러 chunk로 나눌 수 있지만, 원 논문이 single bag label을 다루는 구성과 우리 multi-label 구성은 구분해야 한다. attention weight가 정답 근거나 인과적 설명을 보장하지 않는다.
   https://proceedings.mlr.press/v80/ilse18a.html
4. Guo et al.(ICML2017)은 신경망의 confidence calibration과 temperature scaling을 검토한다. 우리 문제는 여러 독립 sigmoid 라벨이므로 원논문의 multiclass softmax 설정을 그대로 적용하지 않고, held-out BCE/NLL과 reliability를 별도로 평가해야 한다. threshold 조정과 probability calibration은 다른 작업이다.
   https://proceedings.mlr.press/v70/guo17a.html

## 실행 오류도 남기기

MPS SDPA attention의 training dropout 지원 오류를 만나 eager attention으로 전환했다. 실패 로그는 `finetune_first_attempt_mps_dropout_error.log`. 이것은 데이터·모델의 성능 실패와 다른 런타임 제약이다. 무효 실행을 학습 성공 횟수에 포함하지 않는다.

## 아직 확정할 수 없는 것

BERT가 TF-IDF보다 본질적으로 나쁘다 / 생물의학 사전학습이 무의미하다 / 긴 문서라서 낮았다 / anisotropy가 원인이다 / 라벨 관계가 항상 유효하다 / LLM의 인용 문장이 해당 라벨을 의미적으로 입증한다 — 어느 것도 지금 증거만으로 단정하지 않는다.

## 전체 가중치 대신 LoRA를 선택한 이유

성능을 보고 바꾼 것이 아니라 실제 무료16GB Mac에서의 실행 비용을 측정한 뒤 변경했다. 전체 가중치의 batch2 optimizer 업데이트는 약1초였지만 LoRA batch16의 optimizer는 warm-up 후 약.01~.08초였다. forward/backward 변동이 커 전체 학습시간의 선형 배속을 주장하지 않는다. 원본 가중치 고정+저랭크 업데이트는 LoRA 원논문의 접근에서 가져왔고 새로운 알고리즘으로 제안하는 것이 아니다.

https://arxiv.org/abs/2106.09685

본 실행 전 LoRA 학습 장치 배치 오류도 수정했다. 어댑터를 base.weight와 같은 device/dtype으로 생성하고, CPU 경계 검사 및 실제 GPU 단계 검증을 통과했다. 원본 layer와 adapter의 가중치 변화를 따로 검증한다.
