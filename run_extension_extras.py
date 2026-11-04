"""Sequential substantive followups after the main GPU pipeline, then a locked final audit."""

import os, sys, time, json, subprocess, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "extension_v3"
env = dict(os.environ)
env["PYTHONPATH"] = str(ROOT / ".deps") + os.pathsep + env.get("PYTHONPATH", "")


def run(name, script):
    (ROOT / "EXTRAS_STATE.json").write_text(
        json.dumps({"stage": name, "started_monotonic": time.monotonic()}, indent=2)
    )
    print("START", name, flush=True)
    with (V3 / "results" / f"{name}.log").open("w") as output:
        result = subprocess.run(
            [sys.executable, str(V3 / "src" / script)],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect its log")
    print("DONE", name, flush=True)


if __name__ == "__main__":
    while (
        json.loads((ROOT / "PIPELINE_STATE.json").read_text()).get("stage")
        != "extra_core_complete_check_elapsed_and_finalize"
    ):
        time.sleep(5)
    for name, script in [
        ("strong_tfidf", "strong_tfidf.py"),
        ("llm_fewshot", "llm_fewshot.py"),
        ("merge_benchmark", "benchmark_merge.py"),
    ]:
        run(name, script)
    smoke = V3 / "results/service_smoke.json"
    if smoke.exists():
        shutil.copy2(smoke, V3 / "results/service_smoke_unmerged.json")
    run("merged_service_smoke", "smoke_service.py")
    run("chunk_sensitivity", "chunk_sensitivity.py")
    run("bag_mechanisms", "analyze_bags.py")
    run("bag_invariant_tests", "test_bag_invariants.py")
    run("merge_tests", "test_merge.py")
    start = json.loads((V3 / "results/extra_phase_start.json").read_text())
    elapsed = time.monotonic() - start["monotonic_seconds"]
    decision = {
        "elapsed_seconds_before_final_audit": elapsed,
        "replicate512": elapsed < 3601,
        "rule": "Additional seed if substantive extension work is still below60 actual monotonic minutes; not selected from scores.",
    }
    (V3 / "results/time_based_replication_decision.json").write_text(json.dumps(decision, indent=2))
    if decision["replicate512"]:
        run("replica512", "replicate512.py")
    run("adapter_exports", "export_adapters.py")
    run("final_internal_audit", "final_audit.py")
    (ROOT / "EXTRAS_STATE.json").write_text(
        json.dumps(
            {
                "stage": "complete_waiting_final_clock_and_report",
                "elapsed_minutes": (time.monotonic() - start["monotonic_seconds"]) / 60,
            },
            indent=2,
        )
    )
    print("ALL EXTENSION EXPERIMENTS COMPLETE", flush=True)
