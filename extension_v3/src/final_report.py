import sys, os, json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "followup_v2"
V3 = ROOT / "extension_v3"
sys.path.append(str(ROOT / ".deps"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/literature-v2-mpl")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor


def read(p):
    return json.loads(p.read_text())


def kst(value):
    from datetime import datetime, timezone, timedelta

    return (
        datetime.strptime(value, "%Y-%m-%d %H:%M:%S UTC")
        .replace(tzinfo=timezone.utc)
        .astimezone(timezone(timedelta(hours=9)))
        .strftime("%Y-%m-%d %H:%M:%S KST")
    )


def main():
    ft = read(V2 / "results/finetune_results.json")
    inp = read(V2 / "results/input_ablation_results.json")
    diag = read(V2 / "results/diagnostic_results.json")
    v3ft = read(V3 / "results/finetune_results.json")
    if (V3 / "results/replica512_results.json").exists():
        v3ft += read(V3 / "results/replica512_results.json")
    bags = read(V3 / "results/bag_results.json")
    llm = read(V2 / "results/llm_summary.json")
    fewshot = (
        read(V3 / "results/llm_fewshot_summary.json")
        if (V3 / "results/llm_fewshot_summary.json").exists()
        else None
    )
    merge = (
        read(V3 / "results/merge_benchmark.json")
        if (V3 / "results/merge_benchmark.json").exists()
        else None
    )
    sensitivity = (
        read(V3 / "results/chunk_sensitivity.json")
        if (V3 / "results/chunk_sensitivity.json").exists()
        else None
    )
    mechanisms = (
        read(V3 / "results/bag_mechanism_analysis.json")
        if (V3 / "results/bag_mechanism_analysis.json").exists()
        else None
    )
    exports = (
        read(V3 / "results/adapter_exports.json")
        if (V3 / "results/adapter_exports.json").exists()
        else []
    )
    llm_analysis_path = V3 / "results/llm_failure_analysis.json"
    if not llm_analysis_path.exists():
        llm_analysis_path = V3 / "results/llm_zero_failure_analysis.json"
    llm_analysis = read(llm_analysis_path) if llm_analysis_path.exists() else {}
    cal = read(V3 / "results/calibration_results.json")
    old = read(ROOT / "results/test_results.json")
    grid = read(V2 / "results/classifier_grid.json")
    verify = read(V2 / "results/verification.json")
    start = read(V3 / "results/extra_phase_start.json")
    end = read(V3 / "results/extra_phase_end.json")
    allres = diag + inp + ft + v3ft
    if (V3 / "results/strong_tfidf_results.json").exists():
        allres += read(V3 / "results/strong_tfidf_results.json")
    confirmation = []
    if (V3 / "results/final_audit_results.json").exists():
        confirmation = read(V3 / "results/final_audit_results.json")
        allres += confirmation
    confirmation_split = "final_internal_audit1000" if confirmation else "audit"
    rows = []
    for r in old:
        if r["seed"] == 17 and r["method"] in ["biomedbert", "tfidf"]:
            rows.append({"variant": "V1_" + r["method"], "split": "test", **r})
    rows += allres
    agg = []
    for group in ["bag_mean", "uniform_attention", "attention", "label_graph", "random_graph"]:
        for split in (["test", "audit", confirmation_split] if confirmation else ["test", "audit"]):
            rr = [
                r
                for r in bags + confirmation
                if r["variant"].startswith(group + "_seed") and r["split"] == split
            ]
            assert len(rr) == 3
            agg.append(
                {
                    "variant": group + "_3seed_mean",
                    "split": split,
                    "macro_f1": float(np.mean([r["macro_f1"] for r in rr])),
                    "micro_f1": float(np.mean([r["micro_f1"] for r in rr])),
                    "std": float(np.std([r["macro_f1"] for r in rr], ddof=1)),
                }
            )
    figdir = V3 / "results/figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160}
    )
    std = sorted([r for r in grid if r["scaling"] == "standard"], key=lambda r: r["C"])
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot([str(r["C"]) for r in std], [r["train_macro_f1"] for r in std], "-o", label="Train")
    ax.plot([str(r["C"]) for r in std], [r["tune_macro_f1"] for r in std], "-o", label="Tune")
    ax.set_ylim(0, 1)
    ax.set_xlabel("C (smaller = stronger regularization)")
    ax.set_ylabel("Macro-F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "regularization.png")
    plt.close(fig)
    core = ["tuned_frozen256", "last2_seed17", "lora_seed17", "lora_seed42", "lora512_seed17"]
    fig, ax = plt.subplots(figsize=(9, 3))
    positions = np.arange(len(core))
    width = 0.36
    for off, split, color in [
        (-width / 2, "test", "#ed7b24"),
        (width / 2, confirmation_split, "#276bdc"),
    ]:
        vals = [
            next(r["macro_f1"] for r in allres if r["variant"] == n and r["split"] == split)
            for n in core
        ]
        ax.bar(
            positions + off,
            vals,
            width,
            label=(
                "new internal audit"
                if split == "final_internal_audit1000"
                else "reused test" if split == "test" else split
            ),
            color=color,
        )
        for x, v in zip(positions + off, vals):
            ax.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(
        positions,
        [
            "Tuned\nfrozen256",
            "Last 2\nseed17",
            "LoRA256\nseed17",
            "LoRA256\nseed42",
            "LoRA512\nseed17",
        ],
    )
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.set_ylabel("Macro-F1")
    fig.tight_layout()
    fig.savefig(figdir / "finetuning.png")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.76))
    groups = ["bag_mean", "uniform_attention", "attention", "label_graph", "random_graph"]
    labels = ["Bag mean LR", "Uniform head", "Attention", "Label graph", "Random graph"]
    for ax, split in zip(axs, ["test", confirmation_split]):
        rr = [
            next(r for r in agg if r["variant"] == n + "_3seed_mean" and r["split"] == split)
            for n in groups
        ]
        ax.bar(
            labels,
            [r["macro_f1"] for r in rr],
            yerr=[r["std"] for r in rr],
            capsize=4,
            color=["#9099aa", "#778599", "#286bdf", "#ed7b24", "#39a69b"],
        )
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(
            (
                "new internal audit"
                if split == "final_internal_audit1000"
                else "reused test" if split == "test" else split
            )
            + " | mean +/- sample SD"
        )
        ax.set_ylabel("Macro-F1")
    fig.tight_layout()
    fig.savefig(figdir / "bags.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 2.5))
    rr = llm["conditions"] + (fewshot["conditions"] if fewshot else [])
    llm_names = [
        "Zero-shot\nraw",
        "Zero-shot\nquote filter",
        "Frozen\nclassifier",
        "LoRA\nclassifier",
    ] + (["Few-shot\nraw", "Few-shot\nquote filter"] if fewshot else [])
    ax.bar(
        llm_names,
        [r["macro_f1"] for r in rr],
        color=["#9099aa", "#39a69b", "#286bdf", "#ed7b24", "#aa80d0", "#557c55"][: len(rr)],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Same 40 development papers; two topics have zero positives; NOT final test")
    fig.tight_layout()
    fig.savefig(figdir / "llm.png")
    plt.close(fig)
    lines = [
        "# 추가 개발 최종 기록: 진단 → 미세조정 → 긴 문헌 → 검토 서비스",
        "",
        f"V2 완료 후 추가 단계 시작: {kst(start['current_time'])} / 종료: {kst(end['current_time'])} / 실제 경과 {end['elapsed_minutes']:.1f}분. 표시 시각은 사용자 시간대인 KST이며 원시 UTC 기록도 보존했다. 불필요한 대기를 추가하지 않고 실험·검증을 수행했다.",
        "",
        "## 1. 기존 결과 보존과 핵심 원인",
        "V1 baseline을 그대로 보존했다. 최초 frozen BiomedBERT의 저성능을 모델 자체의 한계로 단정하지 않고 분류기 규제·입력 길이·pooling·encoder 적응을 분리했다. C1에서 C.01로 규제를 강하게 하자 train/tune 간격이 줄고, 같은 seed17 재사용 테스트 Macro-F1이 .7484→.7927로 개선됐다. 최초 저성능에 분류기 과적합이 영향을 줬다는 근거다.",
        "",
        "![규제 진단](../results/figures/regularization.png)",
        "",
        "## 2. 실제 결과 전체 비교",
        "| 조건 | 분할 | Macro-F1 | Micro-F1 |",
        "|---|---|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['split']} | {r['macro_f1']:.4f} | {r['micro_f1']:.4f} |"
        )
    for r in agg:
        lines.append(
            f"| {r['variant']} | {r['split']} | {r['macro_f1']:.4f} ± {r['std']:.4f} | {r['micro_f1']:.4f} |"
        )
    lines += [
        "",
        "final_internal_audit1000은 모든 모델·프롬프트를 고정한 뒤 선택한 새 내부1,000편이다. 이 결과로 재튜닝하지 않았으며 모델/정책 해시를 선택 전에 보존했다. V1 3seed 평균과 V2 seed17 한 번의 결과를 섞어 비교하지 않는다. test2,488편은 V1부터 재사용한 개발 평가다. audit1,000편은 V2에서 처음 분리한 내부 확인 표본이며 V3에서는 다시 사용했다. V3를 완전히 새로운 맹검 외부 검증으로 주장하지 않는다.",
        "",
        "## 3. 최종 일회 내부 감사 및 실제 미세조정",
        "V2: 마지막2개 block seed17, 모든 층 LoRA seed17/42. V3: LoRA max512 seed17 추가. partial은 원본 마지막2개 block을, LoRA는 모든층에 추가한 query/value 저랭크 가중치를 task BCE로 업데이트했다. partial3epoch/LoRA1epoch 예산 안에서 tune600으로 checkpoint와 threshold를 선택했고, calibration/test/audit로 고르지 않았다. V2의 처음/마지막 block 가중치 변화는 verification.json에 있다. partial은3epoch/eager/attention dropout.1, LoRA는1epoch/SDPA/attention dropout0이므로 학습량과 구조가 함께 다른 비교다. LoRA256과 LoRA512는 epoch·LR schedule·attention/dropout을 맞췄으나 physical batch와 padding·cache 관리는 다르다. 512 미완료 시도는 보존하고 fixed512 padding/25optimizer step마다 MPS cache 정리를 적용해 재시작했다. 이때 dropout RNG·연산 순서와 호스트 부하도 달라질 수 있다. V3 max512는 읽는 길이와 계산 비용이 함께 증가한다. 하나의 추가 seed 결과이므로 일반적인 우열을 증명하지 않는다.",
        "",
        "![미세조정 비교](../results/figures/finetuning.png)",
        "",
        "## 4. 논문 공부에서 더 확장한 문헌 bag 실험",
        "문헌을 최대4개 token window의 bag으로 표현했다. frozen encoder로 각 조각을 벡터화한 뒤 bag mean LR, 같은 학습 조건의 균등 head, 주제별 attention, 실제 관계 규제, 무작위 관계 규제를 비교했다. 균등 head와 학습 attention은 query의 학습 여부만 다르게 두고 optimizer·dropout·epoch 선택을 맞췄다. 평균 LR과 attention의 차이를 attention 하나의 효과로 해석하지 않는다. 각 조건을3개 학습 부분표본에서 반복했다. 관계 규제는 classifier weight를 가깝게 하는 후속 구성이고, T-MDML의 distance metric/triplet 학습 재현은 아니다.",
        "라벨별 attention은 어떤 조각에 비중을 주었는지 살펴보는 도구다. 정답 근거나 인과적 설명을 보장하지 않는다. 4개보다 많은 window는 균등 선택하여 원문 전체를 항상 읽지는 않는다. 검증 선택 조건이 많아져 선택 과적합 가능성도 남긴다.",
        "",
        "![문헌 bag 비교](../results/figures/bags.png)",
        "",
        "## 5. 생성형 LLM의 실제 효과",
        f"무료 Qwen3-4B-4bit를 로컬 MLX로 실행했다. 고정40편에서 JSON 유효율 {llm['valid_json_rate']:.1%}, 인용 규칙 통과율 {llm['verbatim_evidence_rate']:.1%}, 지연 중앙값 {llm['median_seconds']:.2f}초, p95 {llm['p95_seconds']:.2f}초.",
    ]
    for r in llm["conditions"]:
        lines.append(
            f"- {r['variant']}: Macro-F1 {r['macro_f1']:.4f}, Micro-F1 {r['micro_f1']:.4f}"
        )
    if fewshot:
        lines += [
            "",
            f"train-only 예시 {fewshot['training_examples']}개를 고정한 추가 in-context 대조: JSON {fewshot['valid_json_rate']:.1%}, 인용 규칙 통과 {fewshot['verbatim_evidence_rate']:.1%}, 지연 중앙값 {fewshot['median_seconds']:.2f}초. 같은40편이고 재시도/오류를 본 뒤 예시 선택을 하지 않았다. Qwen 가중치 미세조정은 아니다.",
        ]
        for r in fewshot["conditions"]:
            lines.append(
                f"- {r['variant']}: Macro-F1 {r['macro_f1']:.4f}, Micro-F1 {r['micro_f1']:.4f}"
            )
    lines += [
        "",
        "개발용40편의 탐색 실험이며 최종 test 성능이 아니다. 분류기는4,000편의 과제 학습과 최대256 BERT token, Qwen은 zero/few-shot과 최대1,600 Qwen token의 현재 문헌을 사용했다. 같은 문헌의 pipeline 비교이지 학습·입력·계산 예산을 맞춘 모델 계열 우열 실험은 아니다. 원문에 있는 문장인지 검사해도 의미적 entailment를 검증한 것은 아니다. 분류기 예측을 LLM에 보여주지 않았으며 invalid 출력도 all-negative 오류로 포함했다. LLM의 결과가 낮거나 인용을 잘못 내면 그대로 실패로 기록했다.",
        "",
        "![동일40편 LLM 비교](../results/figures/llm.png)",
        "",
        "## 6. 확률 보정은 정확도와 별도",
        "calibration400으로 scalar temperature를 binary NLL에 맞췄다. threshold도 같은 단조 함수로 변환하여 분류 결과를 유지했다. 따라서 F1은 의도적으로 변하지 않는다. 확률 품질 NLL/Brier/ECE가 개선되는지를 별도로 평가했다.",
    ]
    for r in cal["results"]:
        lines.append(
            f"- {r['split']}: NLL {r['before']['binary_nll']:.4f}→{r['after']['binary_nll']:.4f}, Brier {r['before']['brier_score']:.4f}→{r['after']['brier_score']:.4f}, macro ECE {r['before']['macro_label_ece_10bins']:.4f}→{r['after']['macro_label_ece_10bins']:.4f}; 결정 불변={r['classification_decisions_unchanged']}"
        )
    lines += [
        "",
        "확률 보정이 모든 문헌에서 정확성을 보장하거나 데이터 분포가 바뀌어도 유지된다는 주장은 하지 않는다.",
        "",
        "## 7. 실제 검토 workflow",
        "문헌 내용 해시+모델 checkpoint 버전으로 저장을 구분한다. AI 예측과 사람 검토 라벨은 별도 필드다. SQLite에 저장해 재시작 후 유지하고, 예상 revision이 오래되면409를 반환해 덮어쓰기를 방지한다. 원문에 없는 근거 인용·알 수 없는 라벨을 거절한다. 동일 문헌 중복 등록은 같은 ID를 반환하지만 등록 전 모델 추론 비용까지 중복 제거하는 job queue는 구현하지 않았다.",
        "실제 미세조정 모델을 연결한 API smoke와 DB 동시성·revision·재시작·원문 인용 경계 검사를 수행한다. 검토 입력은 실험용 정답 라벨을 사용한 smoke이며 실제 사람이 운영 환경에서 평가한 결과가 아니다.",
        "",
        "## 8. 제출할 때 설명하기",
        "“처음 결과가 낮게 나온 원인을 분류기 과적합·입력 표현·과제 적응으로 나눠 조사했습니다. 무료 모델을 실제 미세조정하고 기존 결과와 새 내부 감사 결과를 함께 보존했습니다. 논문에서 배운 bag·라벨 관계 개념을 문헌 attention 및 관계 규제로 확장했고, 생성형 LLM 인용 검증과 사람 검토 workflow까지 연결했습니다.”",
        "점수 개선 여부·비용·실패 조건을 위 표에 맞춰 설명한다. 오래 연구한 것처럼 개발 기간을 소급하지 않고, 기존 이론 학습과 이번 구현을 구분한다.",
        "",
        "## 9. 주요 한계",
        "- V1 이후 동일 테스트를 반복 사용했다. V2 audit도 V3에서는 재사용됐다.",
        "- 데이터는 COVID-19 문헌7개 주제이고, 다른 과학 도메인·금융 문서로 일반화하지 않았다.",
        "- 학습 표본4,000편과 제한된 seed/epoch에서의 결과다. SOTA 비교가 아니다.",
        "- LLM40편과 confidence calibration은 소규모 탐색이며 의미적 인용 검증·인간 사용성 시험을 대신하지 않는다.",
        "- 긴 입력과 bag 비교는 compute/input coverage가 달라 완벽한 단일 원인 실험이 아니다.",
        "",
        "## 참고한 원문·공식 문서",
        "- BiomedBERT 모델: https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "- frozen embedding 해석: https://aclanthology.org/2022.findings-emnlp.314/",
        "- 관련 pooling 연구: https://aclanthology.org/2023.emnlp-main.359/",
        "- attention MIL: https://proceedings.mlr.press/v80/ilse18a.html",
        "- attention 설명 평가: https://aclanthology.org/N19-1357/ 및 https://aclanthology.org/D19-1002/",
        "- probability calibration: https://proceedings.mlr.press/v70/guo17a.html",
        "- T-MDML AAAI2025: 사용자가 제공한 Multi-label_AAAI.pdf 원문.",
    ]
    lines += [
        "",
        "## 완료 실행의 학습 비용",
        "| 조건 | 선택 epoch | 학습 실행 분 | 학습 대상 파라미터 | physical batch × accumulation |",
        "|---|---:|---:|---:|---|",
    ]
    training_configs = read(V2 / "results/locked_finetune_policies.json") + read(
        V3 / "results/locked_finetune_policies.json"
    )
    replica_policy = V3 / "results/locked_replica512_policy.json"
    if replica_policy.exists():
        training_configs.append(read(replica_policy))
    for r in training_configs:
        lines.append(
            f"| {r['variant']} | {r['epoch']} | {r['training_seconds']/60:.2f} | {r['trainable_parameters']:,} | {r['physical_batch']} × {r['gradient_accumulation']} |"
        )
    lines += [
        "완료된 각 실행의 학습·epoch 검증·checkpoint 저장 구간이며, 사전 모델 적재·입력 토큰화·최종 test/audit 추론은 별도다. 중단한 실행의 비용은 여기에 합산하지 않고 별도 로그와 전체 개발 경과에 남겼다. 같은 desktop host의 부하가 달랐고 partial3epoch/LoRA1epoch, padding·attention backend도 달라 일반적인 모델 속도 우열로 해석하지 않는다. 512 재시작은 fixed512 padding과25optimizer step마다 cache 정리를 사용했다. host memory/swap 관측도 results에 보존했다."
    ]
    uncertainty_path = V3 / "results/final_audit_uncertainty.json"
    if uncertainty_path.exists():
        uncertainty = read(uncertainty_path)
        lines += [
            "",
            "## 최종 내부 감사의 고정 비교와 불확실성",
            "| 미리 정한 비교 | Macro-F1 차이 | 문헌 paired bootstrap95% 구간 |",
            "|---|---:|---|",
        ]
        for r in uncertainty["comparisons"]:
            lo, hi = r["paired_document_bootstrap_ci95"]
            lines.append(
                f"| {r['comparison']} | {r['delta_macro_f1']:.4f} | [{lo:.4f}, {hi:.4f}] |"
            )
        lines += [
            f"라벨별 support: {uncertainty['label_support']}. 문헌 재표집만의 불확실성이며 모델/설정 선택·도메인 변화는 포함하지 않는다. 다중 비교 보정 없는 서술적 구간이고 전반적 우월성의 검정으로 해석하지 않는다."
        ]
    if llm_analysis:
        lines += [
            "",
            "## LLM 실패를 지표별로 분리",
            "인용 규칙 통과율은 알려진 라벨·15~240자·원문 및 모델에 보인 입력과의 정확 일치를 모두 검사한다. 원래 JSON의 verbatim_evidence_rate 필드는 이 복합 규칙의 통과율이며, 순수한 원문 substring 비율과 구분한다. 불일치는 사실적 환각과 같은 뜻이 아니다.",
        ]
        for condition, item in llm_analysis.items():
            lines += [
                f"- {condition}: 원문 substring 자체의 일치 {item['pure_original_substring_rate']:.1%}; 전체 인용 규칙 통과 {item['full_quote_rule_pass_count']}/{item['items']}. 길이/비일치 사유(중복 가능) {item['invalid_reason_counts_nonexclusive']}. 필터가 제거한 맞는 라벨 {item['filter_removed_true_label_predictions']}개 / 오탐 {item['filter_removed_false_label_predictions']}개."
            ]
        lines += [
            "이40편에는 Transmission·Epidemic Forecasting 정답이0개였다. 전체7라벨 Macro-F1은 이 두 라벨의 F1=0을 포함한다. 모든 조건에 동일한 규칙을 적용했으며, Micro-F1·라벨별 support도 함께 보고한다. 다른 분할의 F1과 직접 비교하지 않는다."
        ]
    if mechanisms:
        lines += [
            "",
            "## 고정 bag 모델의 작동 진단",
            "| 조건 | 선택 lambda | 선택 epoch | audit 정규화 attention entropy | 규제/BCE 비율 |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in mechanisms["models"]:
            lines.append(
                f"| {r['variant']} | {r['selected_lambda']} | {r['selected_epoch']} | {r['mean_normalized_attention_entropy_audit']:.4f} | {r['weighted_penalty_to_train_eval_bce_ratio']:.6g} |"
            )
        lines += [
            "entropy1은 거의 균등,0은 집중이다. lambda0이면 관계 규제를 사용하지 않은 모델이 선택된 것이다. 규제/BCE는 선택 모델 eval-mode의 서술 통계로 확률적 학습 loss와 동일하지 않으며, 작은 값만으로 성능 변화 원인을 증명하지 않는다."
        ]
    if sensitivity:
        lines += [
            "",
            f"문단 제거 민감도: 기존 audit 중 chunk2개 이상 {sensitivity['eligible_documents']}편, 원본 Macro-F1 {sensitivity['baseline']['macro_f1']:.4f}.",
        ]
        for r in sensitivity["conditions"]:
            lines.append(
                f"- {r['removed']}: Macro-F1 {r['macro_f1']:.4f}, 라벨 하나 이상 전환 {r['any_label_flip_fraction']:.1%}, 평균 절대 확률 변화 {r['mean_absolute_probability_change']:.4f}."
            )
        lines += [
            "이는 고정 모델의 masking 민감도이며 원문 인용의 정당성이나 attention의 인과적 설명력을 검증한 것은 아니다. 처음10개 적격 사례를 자동 저장했다."
        ]
    if merge:
        lines += [
            "",
            "## 실제 LoRA 배포 검증",
            "같은32 calibration 문헌·batch8·warm-up·3반복. 고정 순서 실행이므로 일반적인 배속으로 해석하지 않는다.",
            f"최대 확률 차이 {merge['max_probability_difference']:.8g}, 문헌별 라벨집합 일치 {merge['document_labelset_agreement']:.1%}. 처리량과 메모리 원시 수치는 results/merge_benchmark.json에 보존했다. 원본 checkpoint는 수정하지 않았다.",
        ]
    for r in exports:
        lines.append(
            f"- {r['variant']}: adapter+head {r['adapter_bytes']/1e6:.3f}MB / 원본 포함 checkpoint {r['full_checkpoint_bytes']/1e6:.1f}MB. tensor round-trip 정확 일치. 별도 base model이 필요하므로 전체 모델 크기 축소라고 쓰지 않는다."
        )
    (V3 / "docs/REPORT.md").write_text("\n".join(lines))
    out = V3 / "output"
    out.mkdir(exist_ok=True)
    pdfmetrics.registerFont(
        TTFont("Korean", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    )
    pdfmetrics.registerFontFamily(
        "Korean", normal="Korean", bold="Korean", italic="Korean", boldItalic="Korean"
    )
    c = canvas.Canvas(str(out / "portfolio_followup.pdf"), pagesize=(842, 595))
    c.setTitle("BiomedBERT 원인 진단·미세조정·문헌 검토 후속 실험")
    dark = "#142236"
    muted = "#5D6877"
    orange = "#F47C24"

    def text(x, y, s, size=12, color=dark):
        c.setFillColor(HexColor(color))
        c.setFont("Korean", size)
        c.drawString(x, y, s)

    def para(x, y, s, size=11, width=744, color=dark):
        p = Paragraph(
            s,
            ParagraphStyle(
                "p",
                fontName="Korean",
                fontSize=size,
                leading=size * 1.55,
                textColor=HexColor(color),
                wordWrap="CJK",
            ),
        )
        w, h = p.wrap(width, 500)
        p.drawOn(c, x, y - h)
        return y - h

    def head(n, title, sub):
        c.setFillColor(HexColor(orange))
        c.rect(0, 583, 842, 12, fill=1, stroke=0)
        text(42, 550, f"FOLLOW-UP RESEARCH / 2026.11 / {n}", 10, orange)
        text(42, 513, title, 24)
        para(42, 485, sub, 10, color=muted)

    def foot(n):
        text(
            42,
            22,
            "기존 개별연구 → 최초 실패 보존 → 원인 진단 → 무료 로컬 미세조정 → 서비스 검증",
            9,
            muted,
        )
        text(779, 22, str(n), 9, muted)
        text(
            42,
            10,
            "Data: NCBI LitCovid | Models: Microsoft BiomedBERT, Qwen3 | Methods: T-MDML / LoRA / attention MIL (see report)",
            7,
            muted,
        )

    head(
        "01",
        "낮은 점수를 모델의 한계로 단정하지 않았다",
        "V1 결과를 보존하고 분류기 설정·입력 길이·pooling·과제 적응을 분리해 조사했다.",
    )
    c.drawImage(str(figdir / "regularization.png"), 65, 242, width=710, height=231)
    y = 220
    y = (
        para(
            42,
            y,
            "<b>확인한 원인</b>  같은 frozen BiomedBERT 표현에서 C1은 학습 .981 / 검증 .776이었다. 규제를 강화한 C.01은 학습 .869 / 검증 .813으로 간격이 줄었다.",
        )
        - 12
    )
    y = (
        para(
            42,
            y,
            "<b>같은 seed17 테스트 비교</b>  분류기 설정만 조정한 뒤 Macro-F1 .7484 → .7927. 초기 저성능에 과적합이 영향을 줬다는 근거이며 원인 전체를 설명하지는 않는다.",
        )
        - 12
    )
    para(
        42,
        y,
        "<b>추가 확인</b>  256 token으로 제한한 TF-IDF도 full TF-IDF보다 높았다. 따라서 긴 문헌 잘림 하나만을 원인으로 가정하지 않았다. 설정 선택은 검증 데이터에서만 했다.",
        10,
        color=muted,
    )
    foot(1)
    c.showPage()
    head(
        "02",
        "부분·LoRA 미세조정을 별도 감사 표본으로 확인",
        "마지막2개 층 / 모든층 LoRA / 다른 학습 표본 / 긴 입력을 비교. checkpoint는 tune600에서 선택했다.",
    )
    c.drawImage(str(figdir / "finetuning.png"), 45, 219, width=750, height=250)
    best17 = next(r for r in ft if r["variant"] == "lora_seed17" and r["split"] == "test")
    audit17 = next(
        r for r in allres if r["variant"] == "lora_seed17" and r["split"] == confirmation_split
    )
    y = 200
    y = (
        para(
            42,
            y,
            f"<b>미리 지정한 LoRA256 seed17</b>  test Macro-F1 {best17['macro_f1']:.4f} / 내부 audit {audit17['macro_f1']:.4f}. 기존 테스트2,488편과 이전 모델에 쓰지 않은 감사1,000편을 분리해 보고했다.",
        )
        - 12
    )
    y = (
        para(
            42,
            y,
            "<b>실행 증거</b>  학습 곡선·선택 epoch·원본 동결/어댑터 변화·모델 해시를 저장했다. Mac GPU 오류와 실행 프로파일을 남기고 지원 연산·LoRA로 전환한 이유도 기록했다.",
        )
        - 12
    )
    para(
        42,
        y,
        "test는 반복 사용한 개발 평가다. 새 최종 내부 감사1,000편은 모든 모델을 고정한 뒤 한 번 평가하고 재튜닝에 쓰지 않았다. 같은 코퍼스의 내부 확인이며 외부 검증·SOTA로 주장하지 않는다.",
        10,
        color=muted,
    )
    foot(2)
    c.showPage()
    head(
        "03",
        "논문 개념을 긴 문헌의 bag 학습으로 확장",
        "문헌을 최대4개 token window로 나누고 주제별 attention과 관계 규제를 실제로 학습했다.",
    )
    c.drawImage(str(figdir / "bags.png"), 42, 239, width=750, height=230)
    y = 220
    y = (
        para(
            42,
            y,
            "<b>대조 설계</b>  같은 chunk 표현에서 평균 LR / 균등 비중 head / 학습 attention / 실제·무작위 관계 규제를3개 학습 부분표본으로 비교했다.",
        )
        - 12
    )
    y = (
        para(
            42,
            y,
            "<b>원논문과의 경계</b>  bag과 라벨 관계에서 착안한 후속 모델이다. T-MDML의 triplet·전역/개별 거리 행렬을 재현했다고 주장하지 않는다. attention 비중은 정답 근거의 보장이 아니다.",
        )
        - 12
    )
    para(
        42,
        y,
        "<b>설명의 검증</b>  높은/낮은 attention 및 무작위 문단을 제거해 예측 민감도를 비교했다. 모델의 입력 사용을 조사하되 attention을 인과적 근거로 주장하지 않았다. 선택된 관계 규제 강도와 실패 조건도 보존했다.",
        10,
        color=muted,
    )
    foot(3)
    c.showPage()
    head(
        "04",
        "LLM의 설명을 검증하고 사람 검토까지 연결",
        "생성형 Qwen3-4B와 encoder 분류기를 분리해 비교하고, 인용·확률·검토 상태를 각각 검증했다.",
    )
    c.drawImage(str(figdir / "llm.png"), 50, 257, width=742, height=206)
    y = 238
    y = (
        para(
            42,
            y,
            f"<b>무료 로컬 LLM</b>  고정40편의 개발 실험. JSON 유효율 {llm['valid_json_rate']:.1%}, 인용 규칙 통과율 {llm['verbatim_evidence_rate']:.1%}. 인용 일치가 의미적 정답을 증명하지는 않는다.",
        )
        - 10
    )
    y = (
        para(
            42,
            y,
            "<b>확률 보정</b>  별도 calibration400으로 temperature를 선택하고 NLL/Brier/ECE를 평가했다. threshold도 함께 변환해 분류 결정은 유지했다. 정확도와 확률 품질을 구분했다.",
        )
        - 10
    )
    y = (
        para(
            42,
            y,
            "<b>백엔드 검증</b>  문헌 해시·모델 버전으로 등록을 구분하고 AI 예측과 사람 라벨을 분리 저장했다. SQLite 재시작 보존·중복 등록·검토 revision 충돌·원문에 없는 인용을 검사했다.",
        )
        - 10
    )
    para(
        42,
        y,
        "기존 최초 실험·후속 진단·새 모델·실패 결과는 별도 파일로 보존했다. 유료 API를 사용하지 않았다. 전체 수치와 한계는 실험 보고서·raw JSON에서 확인할 수 있다.",
        10,
        color=muted,
    )
    foot(4)
    c.save()
    print("Final report and 4-page portfolio PDF created")


if __name__ == "__main__":
    main()
