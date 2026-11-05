"""Three-page portfolio addendum generated only from saved V4 measurements."""

import json
import os
import sys
from pathlib import Path

from settings import ROOT, V4

sys.path.append(str(ROOT / ".deps"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/literature-v4-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, white


def read(name):
    return json.loads((V4 / "results" / name).read_text())


def main():
    retrieval = read("retrieval_summary.json")["summaries"]
    llm = read("llm_summary.json")["summaries"]
    agent = read("agent_summary.json")
    rewrite = read("rewrite_summary.json")
    recovery = read("quote_recovery.json")
    sql = read("sql_benchmark.json")
    output = V4 / "output/pdf"
    output.mkdir(parents=True, exist_ok=True)
    figpath = V4 / "results/retrieval_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.7), sharey=True)
    for ax, kind, title in zip(
        axes,
        ["lexical", "paraphrase", "korean"],
        ["English keywords (14)", "English paraphrases (14)", "Korean probes (6)"],
    ):
        values = [
            retrieval["test-" + kind][m]["hit5"] for m in ["bm25", "dense", "hybrid"]
        ]
        bars = ax.bar(
            ["BM25", "MiniLM", "Hybrid"],
            values,
            color=["#196c76", "#a9b5c8", "#d58c50"],
        )
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.tick_params(labelsize=9)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.025,
                f"{v:.2f}",
                ha="center",
                fontsize=9,
            )
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Known target Hit@5")
    fig.tight_layout()
    fig.savefig(figpath, dpi=220)
    plt.close(fig)
    pdfmetrics.registerFont(
        TTFont("V4Korean", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    )
    c = canvas.Canvas(
        str(output / "rag_evidence_portfolio.pdf"), pagesize=(595.28, 841.89)
    )
    c.setTitle("Literature Evidence Lab - RAG, MCP and Skill validation")
    c.setAuthor("Sunwoo Park")

    def text(content, y, size=10.8, color="#243349", width=507):
        style = ParagraphStyle(
            "body",
            fontName="V4Korean",
            fontSize=size,
            leading=size * 1.48,
            textColor=HexColor(color),
            wordWrap="CJK",
        )
        p = Paragraph(content, style)
        _, h = p.wrap(width, 1000)
        p.drawOn(c, 44, y - h)
        return y - h - 10

    def header(number, title, subtitle):
        c.setFillColor(HexColor("#196c76"))
        c.rect(0, 818, 595.28, 24, fill=1, stroke=0)
        c.setFillColor(HexColor("#63748a"))
        c.setFont("V4Korean", 9)
        c.drawString(44, 793, "LITERATURE EVIDENCE LAB / V4 / 2026.11.05")
        y = text(title, 770, 22, "#14283b")
        y = text(subtitle, y, 11, "#526478")
        c.setStrokeColor(HexColor("#dce3e9"))
        c.line(44, 44, 551, 44)
        c.setFont("V4Korean", 8)
        c.drawString(
            44, 29, "Public titles/abstracts | Free local models | Preserved failures"
        )
        c.drawRightString(551, 29, f"{number} / 3")
        return y - 8

    def table(rows, y, widths):
        wrapped = [
            [
                Paragraph(
                    str(value),
                    ParagraphStyle(
                        "cell",
                        fontName="V4Korean",
                        fontSize=9.1,
                        leading=13,
                        wordWrap="CJK",
                        textColor=white if i == 0 else HexColor("#243349"),
                    ),
                )
                for value in row
            ]
            for i, row in enumerate(rows)
        ]
        t = Table(wrapped, colWidths=widths)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#196c76")),
                    ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f1f5f8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.4, white),
                ]
            )
        )
        _, h = t.wrap(507, 1000)
        t.drawOn(c, 44, y - h)
        return y - h - 15

    y = header(
        1,
        "문헌 분류에서 검색·근거 검증으로",
        "논문 공부 → 분류 실패 분석 → 검색 방법 대조 → 로컬 LLM·MCP 연결",
    )
    y = text(
        "문제: 문헌을 잘 분류하는 것과 필요한 근거를 검색하는 것은 다르다. 복잡한 모델이 항상 유리한지, 검색된 텍스트에 생성 답변의 근거가 실제로 존재하는지 분리해서 검증했다.",
        y,
    )
    y = text("데이터 파이프라인", y, 14, "#196c76")
    y = text(
        "LitCovid 33,699행 → PMID/내용 해시 중복 43행 제외 → 빈 초록 11행 격리 → 33,645문헌 저장. 500행 단위 커밋과 재실행 시 추가 0건을 확인했다. SQLite FTS5와 읽기 전용 검색 도구를 연결했다.",
        y,
    )
    y = text("같은 문헌에서 5개 검색 방법 비교", y, 14, "#196c76")
    rows = [["방법", "평가 34질의 MRR@10", "평가 Hit@5"]]
    for m, label in [
        ("bm25_raw", "Raw BM25"),
        ("bm25", "BM25 / 운영 기본"),
        ("tfidf", "TF-IDF"),
        ("dense", "MiniLM dense"),
        ("hybrid", "Hybrid RRF"),
    ]:
        r = retrieval["test-all"][m]
        rows.append([label, f"{r['mrr10']:.3f}", f"{r['hit5']:.3f}"])
    y = table(rows, y, [190, 165, 152])
    h = 159
    c.drawImage(str(figpath), 44, y - h, width=507, height=h)
    y -= h + 12
    y = text(
        "개발 12질의 MRR@10으로 BM25를 선택한 뒤 평가 34질의를 열었다. 혼합 검색은 영어 paraphrase Hit@5를 5/14 → 9/14로 높였지만 전체 MRR은 개선하지 못했다. 한국어 6질의는 모든 초기 방법에서 top10 실패였다.",
        y,
        10,
    )
    y = text(
        "범위: AI가 출처를 보고 작성한 known-item 질의다. 다른 관련 논문을 오답으로 판정할 수 없으며 전문가 relevance gold나 운영 검색 일반화 실험은 아니다. MiniLM은 최대 256토큰, lexical 방법은 전체 텍스트를 사용했다.",
        y,
        9,
        "#63748a",
    )
    assert y > 53, y
    c.showPage()
    y = header(
        2,
        "답변과 인용을 따로 검증하기",
        "무료 Qwen3-4B-4bit / 로컬 MLX / temperature 0 / 유료 API 없음",
    )
    y = text(
        "프롬프트를 버전·역할·JSON 출력 스키마로 분리하고 평가 전 해시를 고정했다. 12질문(출처 기반 8 + 범위 밖 4)에서 무문맥, 기본 RAG, 근거 우선 프롬프트를 비교했다. top3 문헌의 처음 1200문자만 보여주고, 실제 보인 구간 안의 연속 인용인지 검사했다.",
        y,
    )
    rows = [
        [
            "조건 / 각 12회",
            "정확 인용 / 생성 인용",
            "지정 출처 인용 / 8",
            "범위 밖 유보 / 4",
        ]
    ]
    for condition, label in [
        ("no-context", "무문맥"),
        ("baseline-v1", "기본 RAG"),
        ("evidence-first-v1", "근거 우선 RAG"),
    ]:
        r = llm[condition]
        rows.append(
            [
                label,
                f"{r['exact_valid_items']} / {r['citation_items']}",
                f"{r['supported_target_cited']} / 8",
                f"{r['unsupported_abstained']} / 4",
            ]
        )
    y = table(rows, y, [153, 133, 112, 109])
    y = text("실패를 지우지 않고 원인을 분리", y, 14, "#196c76")
    y = text(
        "태양 UV 논문에서 모델은 답을 “less than 2 min”이라고 썼지만 인용에 원문에 없는 “...”를 붙였다. 원문 일치 검증은 이를 거절했다. 정답처럼 보이는 답변도 출처 표현이 달라지면 기계적으로 보류될 수 있었다.",
        y,
    )
    y = text(
        f"후속 개선: 앞뒤 생략부호를 제거한 문장이 보이는 원문에서 정확히 한 번 발견될 때만 복구한다. 기존 36회 결과는 그대로 두고 {recovery['new_exact_excerpts']}개 인용을 별도 사후 분석에서 복구했다. 중간 생략·유사 문장·중복 매치는 허용하지 않는다.",
        y,
    )
    y = text("인용 존재 ≠ 주장 검증", y, 14, "#196c76")
    y = text(
        "스타틴 질문에는 서로 다른 메타분석이 답할 수 있었다. 지정 출처와 다른 논문을 인용했다고 바로 환각이라고 볼 수 없다. 따라서 “지정 출처 인용률”과 “의학적 정답률”을 구분했고, 사용자 응답에는 검증된 직접 인용만 표시하도록 했다. 생성된 서술은 미검증 제안으로 보존한다.",
        y,
    )
    y = text("한국어 실패에 대한 추가 진단", y, 14, "#196c76")
    y = text(
        f"고정된 한국어 6질의를 같은 무료 LLM으로 영어 검색어로 바꿔 MCP 검색했다. top10 출처 탐색은 0/6 → {rewrite['rewritten_hit10']}/6이었다. 이미 실패를 확인한 같은 질의에서 한 사후 진단이며, 새 한국어 테스트에서 재현한 성능으로 주장하지 않는다.",
        y,
    )
    strict = read("agent_strict_summary.json")
    y = text(
        f"추가 실패: 숫자 3449를 검색어에 붙여도 OR 검색은 다른 문헌을 허용했고, 무관한 원문 인용이 통과했다. 숫자를 반드시 포함하는 출처 조건을 추가한 사후 재실행에서 지정 출처 인용은 3/4 → {strict['guarded_target_cited']}/4가 됐다. 실패한 soft guard와 원출력도 별도 보존했다.",
        y,
        9.5,
    )
    y = text(
        "근거 우선 프롬프트가 언제나 더 좋은지는 확인하지 못했다. 작은 인공 질문 표본, 문헌 앞부분 제한, 단일 로컬 모델이라는 한계를 원출력·프롬프트 해시·시간과 함께 보존한다.",
        y,
        9,
        "#63748a",
    )
    assert y > 53, y
    c.showPage()
    y = header(
        3,
        "MCP 도구와 재사용 Skill로 연결",
        "재현 가능한 실행·오류 기록·조회 성능까지 프로젝트의 증거로 남기기",
    )
    y = text("실제 연결 구조", y, 14, "#196c76")
    y = text(
        "질문 → 로컬 LLM 계획(JSON) → 도구 allowlist·인자 검사 → stdio MCP 호출 → 근거 추출 → 문맥 일치 검증 → MCP 인용 재검사 → 직접 인용 또는 유보. 한 번의 계획·한 번의 답변 생성과 최대 7회 read-only 도구 호출로 제한했다.",
        y,
    )
    y = table(
        [
            ["구현", "검증 증거"],
            [
                "MCP SDK 2.x 서버",
                "검색 / PMID 조회 / 인용 검증 / corpus 통계, scope resource",
            ],
            [
                "프로토콜 검증",
                "별도 프로세스 initialize·tools/list·call·resource, 오류 포함 12개 통과",
            ],
            [
                "로컬 LLM 도구 선택",
                f"6개 진단 질문, 유효 계획 {agent['valid_plans']}/6, 실제 MCP 호출 {agent['mcp_calls']}회",
            ],
            [
                "프로젝트 Skill",
                "SKILL.md + 실행 helper + UI metadata. 기존 결과 재검증 또는 새 질의 실행",
            ],
        ],
        y,
        [155, 352],
    )
    y = text("DB 쿼리 실험", y, 14, "#196c76")
    indexed = sql["milliseconds"]["indexed_pmid"]["median"]
    scan = sql["milliseconds"]["expression_pmid"]["median"]
    y = text(
        f"동일 문헌 40건을 번갈아 조회했다. PMID 인덱스 쿼리 중앙값 {indexed:.3f} ms, CAST 표현식으로 전체 스캔한 대조 쿼리 {scan:.2f} ms였다. EXPLAIN QUERY PLAN에서 SEARCH와 SCAN을 확인했다. 로컬 warm 조회 실험이며 분산·대규모 운영 처리량은 아니다.",
        y,
    )
    y = text("직접 설명할 수 있는 기여", y, 14, "#196c76")
    y = text(
        "“새 모델을 붙이는 데서 끝내지 않고, 질의를 먼저 고정해 검색 실패를 분리했습니다. 혼합 검색의 순위 손실과 한국어 실패를 기록했고, 프롬프트·인용 검증·도구 호출 경계를 구현했습니다. 실패한 인용은 원본을 남긴 채 정확 매치만 허용하는 복구 옵션으로 확장했습니다.”",
        y,
    )
    y = text(
        "코드·원출력·사전 계획·재검증 helper·커밋을 함께 보존한다. V1–V3의 고정 결과와 모델 해시를 재확인한다. 가중치 미세조정, 전문가 검증, 논문 전문 QA, KG-RAG를 이번 V4에서 했다고 주장하지 않는다.",
        y,
        9,
        "#63748a",
    )
    y = text(
        "주요 출처: NCBI LitCovid 공개 CSV / sentence-transformers/all-MiniLM-L6-v2 모델 카드 / MCP 공식 Python SDK / SQLite FTS5 문서. 상세 링크와 재현 명령은 extension_v4/docs에 제공한다.",
        y,
        8.5,
        "#63748a",
    )
    assert y > 53, y
    c.save()
    print(output / "rag_evidence_portfolio.pdf")


if __name__ == "__main__":
    main()
