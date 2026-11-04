"""Resume only the extension phase; never reruns or overwrites completed V2 results."""

import json
from run_followup_pipeline import ROOT, V3, stage

if __name__ == "__main__":
    assert (V3 / "results/extra_phase_start.json").exists()
    for name, script, log in [
        ("extra_lora512_finetune", V3 / "src/finetune512.py", V3 / "results/finetune512.log"),
        ("extra_chunk_bag_learning", V3 / "src/bag_model.py", V3 / "results/bag.log"),
        ("extra_probability_calibration", V3 / "src/calibrate.py", V3 / "results/calibration.log"),
        (
            "extra_review_store_tests",
            V3 / "src/test_review_store.py",
            V3 / "results/review_store_tests.log",
        ),
        ("extra_actual_review_api", V3 / "src/smoke_service.py", V3 / "results/service_smoke.log"),
    ]:
        stage(name, script, log)
    (ROOT / "PIPELINE_STATE.json").write_text(
        json.dumps({"stage": "extra_core_complete_check_elapsed_and_finalize"}, indent=2)
    )
    print("V3 CORE COMPLETE", flush=True)
