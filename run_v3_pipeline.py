"""Resume one extension stage without retraining completed models or rerunning V2."""

import json, argparse
from run_followup_pipeline import ROOT, V3, stage

STAGES = [
    ("lora512", "extra_lora512_finetune", "finetune512.py", "finetune512.log"),
    ("bag", "extra_chunk_bag_learning", "bag_model.py", "bag.log"),
    ("calibration", "extra_probability_calibration", "calibrate.py", "calibration.log"),
    ("store-tests", "extra_review_store_tests", "test_review_store.py", "review_store_tests.log"),
    ("service", "extra_actual_review_api", "smoke_service.py", "service_smoke.log"),
]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-stage", choices=[s[0] for s in STAGES], default="lora512")
    args = parser.parse_args()
    assert (V3 / "results/extra_phase_start.json").exists()
    start = [s[0] for s in STAGES].index(args.from_stage)
    if start > 0:
        assert (V3 / "results/finetune_results.json").exists() and (
            V3 / "results/locked_finetune_policies.json"
        ).exists()
    for _, name, script, log in STAGES[start:]:
        stage(name, V3 / "src" / script, V3 / "results" / log)
    (ROOT / "PIPELINE_STATE.json").write_text(
        json.dumps({"stage": "extra_core_complete_check_elapsed_and_finalize"}, indent=2)
    )
    print("V3 CORE COMPLETE", flush=True)
