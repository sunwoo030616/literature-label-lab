"""Sequential local GPU jobs; leaves phase milestones for the active assistant."""

import os, sys, time, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "followup_v2"
V3 = ROOT / "extension_v3"
env = dict(os.environ)
env["PYTHONPATH"] = str(ROOT / ".deps") + os.pathsep + env.get("PYTHONPATH", "")


def stage(name, script, log, args=None):
    (ROOT / "PIPELINE_STATE.json").write_text(
        json.dumps(
            {
                "stage": name,
                "script": str(script),
                "started_utc": __import__("datetime")
                .datetime.now(__import__("datetime").timezone.utc)
                .isoformat(),
            },
            indent=2,
        )
    )
    print("START", name, flush=True)
    with (log).open("w") as output:
        proc = subprocess.run(
            [sys.executable, str(script), *(args or [])],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
        )
    if proc.returncode:
        raise RuntimeError(f"{name} failed; see {log}")
    print("DONE", name, flush=True)


if __name__ == "__main__":
    while not (V2 / "results/finetune_results.json").exists():
        log = (
            (V2 / "results/finetune.log").read_text()
            if (V2 / "results/finetune.log").exists()
            else ""
        )
        if "Traceback (most recent call last)" in log:
            raise RuntimeError("Fine-tuning failed; inspect log before proceeding")
        time.sleep(5)
    time.sleep(2)
    for name, script, log in [
        (
            "input_length_and_pooling",
            V2 / "src/input_ablation.py",
            V2 / "results/input_ablation.log",
        ),
        ("generative_llm_evidence", V2 / "src/llm_review.py", V2 / "results/llm.log"),
        ("verify_v2", V2 / "src/verify_v2.py", V2 / "results/verification.log"),
        ("actual_v2_api", V2 / "src/smoke_v2.py", V2 / "results/api_smoke.log"),
        ("report_v2", V2 / "src/report_v2.py", V2 / "results/report.log"),
    ]:
        stage(name, script, log)
    (ROOT / "PIPELINE_STATE.json").write_text(
        json.dumps({"stage": "v2_complete_waiting_extra_phase_clock_anchor"}, indent=2)
    )
    while not (V3 / "results/extra_phase_start.json").exists():
        time.sleep(5)
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
    print("PIPELINE CORE COMPLETE", flush=True)
