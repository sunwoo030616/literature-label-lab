"""Separate JSON validity, quote rules, and annotation errors on the same fixed40."""

import sys, json, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "followup_v2"
V3 = ROOT / "extension_v3"
sys.path.insert(0, str(V2 / "src"))
from common import data, LABELS


def analyze(rows, S):
    reasons = collections.Counter()
    items = 0
    source_exact = 0
    removed_tp = 0
    removed_fp = 0
    examples = []
    for row in rows:
        doc = S["dev"][row["index"]]["text"]
        for item in row["items"]:
            items += 1
            quote = item["evidence"]
            flags = []
            if item["label"] not in LABELS:
                flags.append("unknown_label")
            if not isinstance(quote, str):
                flags.append("not_a_string")
            else:
                source_exact += int(quote in doc)
                if not 15 <= len(quote) <= 240:
                    flags.append("quote_length_outside_15_240")
                if quote not in doc:
                    flags.append("not_verbatim_in_original")
                if quote in doc and not flags and not item["exact_quote_valid"]:
                    flags.append("outside_model_visible_input")
            if not item["exact_quote_valid"]:
                reasons.update(flags or ["other_validation_failure"])
                examples.append(
                    {
                        "pmid": row["pmid"],
                        "label": item["label"],
                        "evidence": quote,
                        "reasons": flags,
                        "gold": row["gold"],
                    }
                )
        removed = set(row["raw_topics"]) - set(row["verified_topics"])
        removed_tp += len(removed & set(row["gold"]))
        removed_fp += len(removed - set(row["gold"]))
    perlabel = {}
    for label in LABELS:
        gold = sum(label in row["gold"] for row in rows)
        raw = sum(label in row["raw_topics"] for row in rows)
        tp = sum(label in row["raw_topics"] and label in row["gold"] for row in rows)
        perlabel[label] = {
            "gold_support": gold,
            "raw_predicted": raw,
            "raw_true_positive": tp,
            "raw_false_positive": raw - tp,
            "raw_false_negative": gold - tp,
        }
    return {
        "documents": len(rows),
        "items": items,
        "exact_substring_in_original_count": source_exact,
        "pure_original_substring_rate": source_exact / items if items else None,
        "full_quote_rule_pass_count": sum(
            item["exact_quote_valid"] for row in rows for item in row["items"]
        ),
        "full_quote_rule_definition": "known label, string length15..240, exact original substring AND exact model-visible substring",
        "invalid_reason_counts_nonexclusive": dict(reasons),
        "filter_removed_true_label_predictions": removed_tp,
        "filter_removed_false_label_predictions": removed_fp,
        "per_label": perlabel,
        "topics_with_no_positive_examples": [
            k for k, v in perlabel.items() if v["gold_support"] == 0
        ],
        "invalid_examples": examples,
        "note": "String occurrence does not establish semantic support. Length violations are not automatically fabricated quotations. Full-rule rate was named verbatim_evidence_rate in the original output; preserve that field and use this decomposition to interpret it.",
    }


if __name__ == "__main__":
    S, _, _, _ = data()
    result = {
        "zero_shot": analyze(json.loads((V2 / "results/llm_raw_results.json").read_text()), S)
    }
    few = V3 / "results/llm_fewshot_raw.json"
    if few.exists():
        rows = json.loads(few.read_text())
        assert len(rows) == 40
        result["few_shot"] = analyze(rows, S)
    name = "llm_failure_analysis.json" if "few_shot" in result else "llm_zero_failure_analysis.json"
    (V3 / "results" / name).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        {
            k: {
                f: v
                for f, v in a.items()
                if f
                in [
                    "pure_original_substring_rate",
                    "full_quote_rule_pass_count",
                    "invalid_reason_counts_nonexclusive",
                    "filter_removed_true_label_predictions",
                    "filter_removed_false_label_predictions",
                ]
            }
            for k, a in result.items()
        }
    )
