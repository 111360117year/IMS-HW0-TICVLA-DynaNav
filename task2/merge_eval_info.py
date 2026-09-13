#!/usr/bin/env python
"""Merge several per-suite eval_info.json files (produced by eval_libero_liveview.py) into one.

Usage:
  python merge_eval_info.py --out eval_info.json outputs/eval/libero_spatial/eval_info.json \
      outputs/eval/libero_object/eval_info.json outputs/eval/libero_goal/eval_info.json \
      outputs/eval/libero_10/eval_info.json

Prints the per-suite table with the SmolVLA paper targets (+/- 3 pp) next to it.
"""

import argparse
import json
from pathlib import Path

PAPER = {"libero_spatial": 90.0, "libero_object": 96.0, "libero_goal": 92.0, "libero_10": 71.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="eval_info.json")
    args = ap.parse_args()

    per_task, per_group, videos = [], {}, []
    succ, sum_r, max_r, eval_s = [], [], [], 0.0
    metas = []
    for p in args.inputs:
        d = json.load(open(p))
        per_task.extend(d["per_task"])
        per_group.update(d["per_group"])
        videos.extend(d["overall"].get("video_paths", []))
        eval_s += d["overall"].get("eval_s", 0.0)
        for t in d["per_task"]:
            succ.extend(t["metrics"]["successes"])
            sum_r.extend(t["metrics"]["sum_rewards"])
            max_r.extend(t["metrics"]["max_rewards"])
        metas.append(d.get("meta", {}))

    n = len(succ)
    merged = {
        "per_task": per_task,
        "per_group": per_group,
        "overall": {
            "avg_sum_reward": sum(sum_r) / n if n else float("nan"),
            "avg_max_reward": sum(max_r) / n if n else float("nan"),
            "pc_success": 100.0 * sum(succ) / n if n else float("nan"),
            "n_episodes": n,
            "eval_s": eval_s,
            "eval_ep_s": eval_s / n if n else float("nan"),
            "video_paths": videos,
        },
        "meta": {"merged_from": [str(Path(p)) for p in args.inputs], "sources": metas},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(merged, open(args.out, "w"), indent=2)

    print(f"{'suite':<16}{'episodes':>10}{'ours %':>10}{'paper %':>10}{'diff':>8}  within +/-3pp")
    for suite, g in per_group.items():
        paper = PAPER.get(suite)
        diff = g["pc_success"] - paper if paper is not None else float("nan")
        ok = "yes" if paper is not None and abs(diff) <= 3.0 else "NO"
        print(f"{suite:<16}{g['n_episodes']:>10}{g['pc_success']:>10.1f}{paper if paper is not None else float('nan'):>10.1f}{diff:>+8.1f}  {ok}")
    avg_paper = sum(PAPER[s] for s in per_group if s in PAPER) / max(1, len([s for s in per_group if s in PAPER]))
    print(f"{'overall':<16}{n:>10}{merged['overall']['pc_success']:>10.1f}{avg_paper:>10.1f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
