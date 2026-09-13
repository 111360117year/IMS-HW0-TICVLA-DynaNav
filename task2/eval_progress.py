#!/usr/bin/env python
"""Show the progress and results of a running or finished run_eval_all.sh evaluation.

Usage:  python eval_progress.py <output_dir>        e.g. python eval_progress.py $HW0/outputs/eval_final

Reads the per-suite eval_info.json files (written after every task) and counts the recorded
episode videos of the task in progress, so it works while the evaluation is still running.
"""

import json
import sys
from pathlib import Path

PAPER = {"libero_spatial": 90.0, "libero_object": 96.0, "libero_goal": 92.0, "libero_10": 71.0}
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval_final")
    total_succ = total_done = 0
    for suite in SUITES:
        info_path = out / suite / "eval_info.json"
        videos = sorted((out / suite / "videos" / suite).glob("*.mp4")) if (out / suite / "videos" / suite).exists() else []
        v_succ = sum(1 for v in videos if v.stem.endswith("_SUCCESS"))
        if not info_path.exists() and not videos:
            print(f"== {suite:<15} not started")
            continue
        per_task = json.load(open(info_path))["per_task"] if info_path.exists() else []
        done_tasks = {t["task_id"] for t in per_task}
        n_task_eps = sum(len(t["metrics"]["successes"]) for t in per_task)
        succ_tasks = sum(sum(t["metrics"]["successes"]) for t in per_task)
        # episodes of the task currently in progress = videos not yet accounted for in eval_info.json
        in_progress_done = len(videos) - n_task_eps
        in_progress_succ = v_succ - succ_tasks
        done, succ = len(videos), v_succ
        total_done += done
        total_succ += succ
        rate = 100.0 * succ / done if done else float("nan")
        state = "finished" if len(done_tasks) >= 10 else f"running (task {len(done_tasks) + 1}/10, ep {in_progress_done}/10, {in_progress_succ} ok)"
        print(f"== {suite:<15} {succ:3d}/{done:<3d} = {rate:5.1f}%   paper {PAPER[suite]:.0f}   {state}")
        for t in per_task:
            m = t["metrics"]
            s, n = sum(m["successes"]), len(m["successes"])
            flag = "" if s >= 8 else "  <-- weak"
            print(f"   task {t['task_id']:2d}  {s:2d}/{n:<2d}  steps {min(m['n_steps']):3d}-{max(m['n_steps']):3d}  {t['task_name'][:58]}{flag}")
    if total_done:
        print(f"\nTOTAL {total_succ}/{total_done} = {100.0 * total_succ / total_done:.1f}%   ({total_done}/400 episodes done)")


if __name__ == "__main__":
    main()
