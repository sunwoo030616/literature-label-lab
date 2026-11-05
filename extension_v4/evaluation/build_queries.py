"""AI-authored known-item queries, specified before observing search outputs."""

import hashlib
import json
import sqlite3
from pathlib import Path

V4 = Path(__file__).resolve().parents[1]
# Source documents selected by deterministic hash, not retrieval score.
PAIRS = [
    (
        "Ramazzini occupational infection prevention employer government responsibilities",
        "Which statement asks public authorities and businesses to protect workers exposed to infected people?",
    ),
    (
        "ACE2 angiotensin receptor blockers inhibitors susceptibility evidence",
        "Should medicines for blood pressure be stopped because they may raise the coronavirus entry receptor?",
    ),
    (
        "radiopharmaceutical access availability 35 countries nuclear medicine supply",
        "What international survey describes shortages of radioactive medicines and dependence on a single supplier?",
    ),
    (
        "agranulocytosis COVID recovery TBO filgrastim thrombocytosis",
        "Find the case of a man whose neutrophils disappeared after recovering from coronavirus and returned after growth factor treatment.",
    ),
    (
        "plasminogen tPA inflammation efferocytosis M2 macrophages",
        "Which review examines clot dissolving enzymes in immune signaling and clearance of dead cells?",
    ),
    (
        "neurosurgical residency New York census 72 operations resident deployment",
        "How did a surgical training program in the outbreak epicenter measure lost procedures and trainee sick days?",
    ),
    (
        "dexamethasone pro resolving lipid mediators cytokine storms",
        "Find a discussion linking an anti-inflammatory steroid, inflammation resolution and the sudden excessive immune response in coronavirus disease.",
    ),
    (
        "statin meta analysis 9 studies 3449 in hospital mortality",
        "Did pooled evidence show that cholesterol lowering medicines helped people hospitalized with coronavirus infection?",
    ),
    (
        "Buenos Aires private institution 26 cases fever March 2020",
        "Which early Argentine hospital series described two fever patterns among mostly imported infections?",
    ),
    (
        "chest CT 101 Hunan ground glass traction bronchiectasis emergency",
        "What four-center study connects lung scan abnormalities with mild versus severe illness in one hundred and one patients?",
    ),
    (
        "CONTROL 19 Italian rheumatology registry 232 immunomodulatory",
        "What did the first Italian surveillance results say about outcomes for people with autoimmune joint disorders taking immune modifying drugs?",
    ),
    (
        "acute transverse myelitis 28 female plasma exchange urinary retention",
        "Find the young woman with trouble urinating and symmetric limb numbness after infection who improved with steroids and blood plasma replacement.",
    ),
    (
        "spike N terminal domain S1 NTD ganglioside druggable pockets",
        "Which perspective proposes a viral surface protein region outside the receptor binding domain as a small molecule attachment blocking target?",
    ),
    (
        "solar UV B A inactivation 63 percent two minutes Solar Pump",
        "Find the sunlight study estimating how quickly summer noon exposure disables nearly two thirds of airborne virus particles.",
    ),
    (
        "maternal child health services low middle income disasters systematic review twenty",
        "Which evidence synthesis describes disrupted access to care for mothers and children during infectious outbreaks and disasters in poorer countries?",
    ),
    (
        "serological ELISA IgM IgG N protein 238 hospital patients 11 days",
        "Which hospital study compared blood antibody detection with throat swab RNA tests and found their usefulness changed with time since symptoms began?",
    ),
    (
        "psychopharmacology COVID psychotropics pharmacokinetic interactions organ systems",
        "What review warns that infection related organ changes and experimental antiviral treatments can alter psychiatric medicine safety?",
    ),
    (
        "differential ventilation flow control valves two pairs lung compliance",
        "Can one breathing machine support two people with different lung stiffness using separate valves, and what small experiment tested this?",
    ),
    (
        "critically ill ICU therapeutic strategies respiratory circulation nutrition",
        "Find an early overview combining breathing support, circulation monitoring and nutrition for intensive care before specific vaccines were available.",
    ),
    (
        "radiographic 240 patients chest X ray symptom onset negative 25 percent",
        "Which study of two hundred and forty infected people reports how ordinary lung X ray abnormalities change with days after symptoms?",
    ),
]
KOREAN = {
    7: "코로나 입원 환자에게 스타틴이 도움이 되었는지 9개 연구 3449명을 분석한 논문",
    9: "후난성 네 기관의 101명 환자에서 흉부 CT 소견과 중증도를 비교한 연구",
    11: "감염 후 배뇨 곤란과 팔다리 감각 저하를 보인 28세 여성 횡단성 척수염 증례",
    13: "여름 정오의 태양 자외선으로 바이러스 63퍼센트를 비활성화하는 데 걸리는 시간 연구",
    15: "입원 환자 238명에서 증상 시작 이후 시간에 따른 항체 검사와 핵산 검사를 비교한 연구",
    17: "폐의 탄성이 다른 두 환자를 유량 조절 밸브로 한 인공호흡기에 연결한 실험",
}


def main():
    c = sqlite3.connect(V4 / "data/corpus.sqlite3")
    c.row_factory = sqlite3.Row
    docs = sorted(
        c.execute("SELECT pmid,title,abstract,content_hash FROM documents"),
        key=lambda r: hashlib.sha256(
            ("v4-known-item-20261105" + r["pmid"]).encode()
        ).hexdigest(),
    )[:20]
    queries = []
    for i, (row, pair) in enumerate(zip(docs, PAIRS, strict=True)):
        for kind, text in zip(["lexical", "paraphrase"], pair):
            queries.append(
                dict(
                    id=f"q{i:02d}-{kind}",
                    source_group=i,
                    split="dev" if i < 6 else "test",
                    kind=kind,
                    query=text,
                    target_pmid=row["pmid"],
                    target_content_hash=row["content_hash"],
                )
            )
        if i in KOREAN:
            queries.append(
                dict(
                    id=f"q{i:02d}-korean",
                    source_group=i,
                    split="test",
                    kind="korean",
                    query=KOREAN[i],
                    target_pmid=row["pmid"],
                    target_content_hash=row["content_hash"],
                )
            )
    (V4 / "evaluation/queries.json").write_text(
        json.dumps(
            dict(
                author="AI assistant, manually paraphrased from source abstracts; no independent expert judgments",
                selection="20 smallest SHA256(v4-known-item-20261105 + PMID)",
                queries=queries,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print("wrote", len(queries), "queries; no retrieval run")


if __name__ == "__main__":
    main()
