import json, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "followup_v2"
import numpy as np


def read(n):
    return json.loads((V2 / "results" / n).read_text())


def main():
    ft = read("finetune_results.json")
    diag = read("diagnostic_results.json")
    inp = read("input_ablation_results.json")
    llm = read("llm_summary.json")
    cfg = read("locked_finetune_policies.json")
    verify = read("verification.json")
    grid = read("classifier_grid.json")
    lengths = read("length_audit.json")
    os.environ.setdefault("MPLCONFIGDIR", str(V2 / "data/mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = V2 / "results/figures"
    figdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for policy in cfg:
        history = read(policy["variant"] + "_history.json")
        axes[0].plot(
            [r["epoch"] for r in history],
            [r["train_bce"] for r in history],
            "-o",
            label=policy["variant"],
        )
        axes[1].plot(
            [r["epoch"] for r in history],
            [r["tune"]["macro_f1"] for r in history],
            "-o",
            label=policy["variant"],
        )
    axes[0].set_title("Training BCE")
    axes[1].set_title("Tune Macro-F1")
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Unequal budgets: last2 3 epochs; LoRA 1 epoch. No test-based selection.")
    fig.tight_layout()
    fig.savefig(figdir / "learning_curves.png", dpi=180)
    plt.close(fig)
    allres = diag + inp + ft
    old = json.loads((ROOT / "results/test_results.json").read_text())
    baseline = next(r for r in old if r["seed"] == 17 and r["method"] == "biomedbert")
    tf = next(r for r in old if r["seed"] == 17 and r["method"] == "tfidf")
    lines = [
        "# V2: BiomedBERT 저성능 원인 진단과 무료 미세조정",
        "",
        "V1 최초 실험을 보존한 후속 개발. 재사용 테스트와 새 내부 감사 표본을 구분한다.",
        "",
        "## 1. 이미 확인한 원인: 분류기 과적합",
        "같은 frozen256 표현과 학습 표본에서 규제를 바꿨다. StandardScaler+C1은 train Macro-F1 .981 / tune .776, C.01은 train .869 / tune .813이었다. tune으로 고른 C.01의 기존 test 결과는 .7927로 V1 seed17 .7484보다 높았다. 모델 표현을 바꾸지 않고도 개선돼, 최초 저성능의 일부가 분류기 설정에 기인한다는 근거다. 원인 전체를 설명하거나 독립 외부 재현을 의미하지는 않는다.",
        "",
        "## 2. 입력 길이와 pooling",
        "| 분할 | 길이 중앙값 | 256 초과 | 512 초과 |",
        "|---|---:|---:|---:|",
    ]
    for k, r in lengths.items():
        lines.append(f"| {k} | {r['median']:.0f} | {r['over256']:.1%} | {r['over512']:.1%} |")
    lines += [
        "",
        "TF-IDF 256제한은 BiomedBERT tokenizer로 잘라 단어를 복원했다. 잘림뿐 아니라 vocabulary 구성이 변하므로 순수한 길이 효과라고 단정하지 않는다. 512 mean/CLS는 같은 forward에서 계산해 그 둘의 pooling 비교는 길이를 맞췄다.",
        "",
        "## 3. 최종 결과 비교",
        "| 모델 | 평가 분할 | Macro-F1 | Micro-F1 | Hamming loss |",
        "|---|---|---:|---:|---:|",
        f"| V1 frozen256 seed17 | reused test | {baseline['macro_f1']:.4f} | {baseline['micro_f1']:.4f} | {baseline['hamming_loss']:.4f} |",
        f"| V1 TF-IDF seed17 | reused test | {tf['macro_f1']:.4f} | {tf['micro_f1']:.4f} | {tf['hamming_loss']:.4f} |",
    ]
    for r in allres:
        lines.append(
            f"| {r['variant']} | {r['split']} | {r['macro_f1']:.4f} | {r['micro_f1']:.4f} | {r['hamming_loss']:.4f} |"
        )
    lines += [
        "",
        "test는 이미 V1에서 본 2,488편이다. audit는 기존 선택/학습에 쓰지 않은 공식 train 문헌1,000편이다. 감사 표본의 라벨로 설정을 선택하지 않았다. 동일 데이터 원천의 내부 IID 표본이며 완전히 새로운 외부 검증은 아니다. 서로 다른 seed의 학습 표본도 달라 점수 차이를 단순한 초기화 효과로 해석하지 않는다.",
        "",
        "## 4. 실제 가중치 미세조정",
        "![학습과 검증 곡선](../results/figures/learning_curves.png)",
        "mean pooling+dropout+linear multi-label head, BCEWithLogitsLoss, partial encoder LR2e-5 / LoRA LR5e-4 / head1e-3, effective batch16, max256, partial3epoch/LoRA1epoch. 마지막2개 block의 원본 가중치를 학습한 조건과 모든12개 block query/value에 LoRA를 붙인 seed17/42를 비교했다. 원본 전체 가중치 학습은 비용 때문에 중단했으며 성공 결과로 세지 않는다. 모든 checkpoint/threshold 선택은 tune600에서만 수행했다.",
    ]
    for c in cfg:
        lines.append(
            f"- {c['variant']}: 선택 epoch {c['epoch']}, tune Macro-F1 {c['tune_macro_f1']:.4f}, requires_grad 파라미터 {c['trainable_parameters']:,}, 학습 {c['training_seconds']/60:.1f}분. requires_grad 수는 optimizer 대상이며 사용하지 않는 pooler는 실제 gradient를 받지 않을 수 있다."
        )
    lines += ["", "가중치 변화 검사:"]
    for c in verify["actual_encoder_weight_updates"]:
        lines.append(
            f"- {c['variant']}: 마지막 block 변화 L2={c['last_block_delta_l2']:.6f}, 첫 block={c['first_block_delta_l2']:.6f}, LoRA B norm 합={c.get('adapter_B_l2_sum',0):.6f}, 원본 변경 텐서={c.get('base_tensors_changed',0)}개."
        )
    lines += [
        "",
        "MPS의 SDPA dropout 미지원 오류를 보존하고 eager attention으로 재시작했다. 실패 실행을 성공 학습으로 세지 않았다. partial은3epoch/eager/attention dropout.1, LoRA는1epoch/SDPA/attention dropout0을 사용하므로 학습량과 층 수만의 순수 비교가 아니다. 원래 masked-LM head가 남긴 unexpected key는 mean-pooled encoder로 다른 task를 만들 때 사용하지 않는 출력 head이며, 실제 encoder 가중치는 사전학습 상태로 적재했다.",
        "",
        "## 5. 생성형 LLM을 덧붙인 실제 검증",
        f"무료 로컬 {llm['model']}. calibration400 중 문헌 해시로 미리 고정한40편에 zero-shot 주제·원문 인용을 요청했다. 분류기 예측을 보여주지 않았다. JSON 유효율 {llm['valid_json_rate']:.1%}, 출력 인용의 인용 규칙 통과율 {llm['verbatim_evidence_rate']:.1%}, 지연 중앙값 {llm['median_seconds']:.2f}초/p95 {llm['p95_seconds']:.2f}초.",
        "| 방식 | 같은40편 Macro-F1 | Micro-F1 |",
        "|---|---:|---:|",
    ]
    for r in llm["conditions"]:
        lines.append(f"| {r['variant']} | {r['macro_f1']:.4f} | {r['micro_f1']:.4f} |")
    lines += [
        "",
        "이40편은 개발용 calibration의 일부라 최종 테스트 성능으로 쓰지 않는다. 분류기는4,000편 과제 학습/최대256 BERT token, Qwen은 zero-shot/최대1,600 Qwen token을 사용한 pipeline 비교다. 학습·입력 예산을 맞춘 모델 계열 우열 실험이 아니다. Transmission·Epidemic Forecasting 정답이0개라 전체7라벨 Macro-F1은 이 두 라벨의0점을 포함하며 같은40편 조건끼리 비교한다. invalid/empty 출력은 all-negative로 평가했다. 원래 JSON의 verbatim_evidence_rate는 알려진 라벨·15~240자·원문 및 모델 입력과의 정확 일치라는 복합 인용 규칙의 통과율이다. 순수한 원문 substring 비율과 다르며, 길이 위반을 사실적 환각으로 부르지 않는다. 인용 원문 일치는 의미적 entailment나 모델 설명의 충실성을 보장하지 않는다. JSON parsing만 통과한 출력과 근거 검증 후 출력을 구분했다.",
        "",
        "## 6. 원인별로 말할 수 있는 범위",
        "- 규제: 같은 표현에서 train/tune 간격이 줄고 후속 평가가 개선됐다. 분류기 과적합이 최초 결과에 영향을 줬다는 직접 근거다.",
        "- 입력 길이: 256/512과 matched pooling의 결과로 판단한다. 길게 읽으면 항상 좋아진다고 가정하지 않는다.",
        "- 과제 적응: 실제 encoder weight update 및 frozen 대비 결과를 확인한다. fine-tuning의 optimizer/head/dropout도 함께 달라지므로 표현 변화 하나만의 순수한 인과 효과라고 과장하지 않는다.",
        "- 생성형 LLM: 원문 근거 추출 도구로 쓸 가능성을 소규모 개발 실험으로 확인한다. LLM을 붙였다는 사실만으로 정확성이 개선된다고 쓰지 않는다.",
        "",
        "## 7. 보존·재현",
        f"V1 {verify['v1_files_unchanged']}개 파일의 SHA256이 실행 전과 동일하다. 원본 결과·PDF·모델을 덮어쓰지 않았다. 감사 표본의 PMID/텍스트 해시는 기존 train/dev/test와 겹치지 않는다.",
        "실행 순서: diagnose.py → finetune.py → input_ablation.py → llm_review.py → verify_v2.py → report_v2.py. GPU 모델은 순차 실행했다. V2 inference: PYTHONPATH=followup_v2/src uvicorn api_v2:app --host 127.0.0.1 --port8768.",
        "",
        "## 8. 설명 문장",
        "“처음 frozen BiomedBERT가 TF-IDF보다 낮게 나왔지만 모델 자체의 한계로 단정하지 않고, 규제·입력 길이·pooling·과제 적응을 분리해 조사했습니다. 기존 결과를 보존하고 무료 로컬 미세조정 및 별도 감사 평가를 수행했으며, 생성형 LLM의 주제·근거 추출도 같은 문헌에서 비교했습니다.”",
        "",
        "수치는 결과표에 맞춰 추가하고, 본인이 실제로 이해·검증한 범위로 설명한다.",
        "",
        "## 출처",
        "- 모델: https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "- text classification fine-tuning 공식 문서: https://huggingface.co/docs/transformers/tasks/sequence_classification",
        "- frozen embedding 해석의 주의점: https://aclanthology.org/2022.findings-emnlp.314/",
        "- 관련 pooling 연구: https://aclanthology.org/2023.emnlp-main.359/",
    ]
    errors = V2 / "results/finetuning_error_analysis.json"
    if errors.exists():
        analysis = json.loads(errors.read_text())
        lines += [
            "",
            "## 고정 모델의 paired bootstrap 및 오류 전환",
            "| 분할 | 비교 | Macro-F1 차이 | 문헌 bootstrap95% 구간 | 완전정답으로 수정 | 기존 완전정답 손상 |",
            "|---|---|---:|---|---:|---:|",
        ]
        for split, comparisons in analysis["comparisons"].items():
            for name, item in comparisons.items():
                lo, hi = item["bootstrap_ci95"]
                lines.append(
                    f"| {split} | {name} | {item['point_delta_macro_f1']:.4f} | [{lo:.4f}, {hi:.4f}] | {item['exact_match_fixed']} | {item['exact_match_broken']} |"
                )
        lines += [
            "문헌 재표집에 따른 불확실성만 나타내며, 설정 선택·학습 표본·새 도메인에 대한 불확실성을 포함하지 않는다. 사례는 고정 순서의 최초 오류를 자동 선택했다. 라벨별 FP/FN과 실제 사례는 finetuning_error_analysis.json에서 확인한다."
        ]
        for example in analysis["examples"][:4]:
            lines.append(
                f"- PMID {example['pmid']} / {example['variant']} / {example['split']}: 정답 {example['gold']}, 예측 {example['predicted']}, 누락 {example['false_negative']}, 오탐 {example['false_positive']}."
            )
    (V2 / "docs/REPORT.md").write_text("\n".join(lines))
    print("V2 report created")


if __name__ == "__main__":
    main()
