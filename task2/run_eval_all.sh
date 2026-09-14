#!/usr/bin/env bash
# Evaluate one checkpoint on the four standard LIBERO suites (10 tasks x 10 episodes each = 400 episodes)
# with the live-view evaluator, then merge the four eval_info.json files into one.
#
# Usage:
#   ./run_eval_all.sh <pretrained_model_dir> <output_dir> [gpu_list]
#
#   env vars  N_EPISODES (default 10), N_ACTION_STEPS (default 10), SEED (default 1000), PORT_BASE (default 8765)
#   gpu_list  comma-separated GPU ids. One id  -> the four suites run one after another on that GPU.
#             Four ids -> each suite runs on its own GPU in parallel (about 2.5 h instead of 5 h).
#
# Examples:
#   ./run_eval_all.sh $HW0/outputs/train_smolvla_libero/checkpoints/100000/pretrained_model $HW0/outputs/eval_final 3
#   ./run_eval_all.sh $HW0/outputs/train_smolvla_libero/checkpoints/100000/pretrained_model $HW0/outputs/eval_final 0,1,2,3
set -euo pipefail

CKPT=${1:?pretrained_model dir}
OUT=${2:?output dir}
GPUS=${3:-0}
N_EPISODES=${N_EPISODES:-10}
N_ACTION_STEPS=${N_ACTION_STEPS:-10}   # actions executed per policy call (paper: 1 or 10 for LIBERO)
SEED=${SEED:-1000}                     # evaluation seed (LeRobot default 1000)
PORT_BASE=${PORT_BASE:-8765}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SUITES=(libero_spatial libero_object libero_goal libero_10)    # LeRobot env ids
NAMES=(libero_spatial libero_object libero_goal libero_long)   # names of our output folders / logs
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
mkdir -p "$OUT"

run_suite() {  # env_id gpu port name
  CUDA_VISIBLE_DEVICES="$2" python -u "$HERE/eval_libero_liveview.py" \
    --policy.path="$CKPT" \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --env.type=libero --env.task="$1" \
    --env.observation_height=256 --env.observation_width=256 \
    --eval.n_episodes="$N_EPISODES" --eval.batch_size=1 \
    --seed="$SEED" \
    --output_dir="$OUT/$4" \
    --liveview.port="$3" \
    > "$OUT/$4.log" 2>&1
}

if [ "${#GPU_ARR[@]}" -ge 4 ]; then
  echo "Running the four suites in parallel on GPUs ${GPUS} (live views on ports ${PORT_BASE}..$((PORT_BASE+3)))"
  pids=()
  for i in "${!SUITES[@]}"; do
    run_suite "${SUITES[$i]}" "${GPU_ARR[$i]}" $((PORT_BASE + i)) "${NAMES[$i]}" &
    pids+=($!)
    echo "  ${NAMES[$i]} -> GPU ${GPU_ARR[$i]}, port $((PORT_BASE + i)), pid ${pids[-1]}, log $OUT/${NAMES[$i]}.log"
  done
  for p in "${pids[@]}"; do wait "$p"; done
else
  echo "Running the four suites sequentially on GPU ${GPU_ARR[0]} (live view on port ${PORT_BASE})"
  for i in "${!SUITES[@]}"; do
    echo "  ${NAMES[$i]} ... (log $OUT/${NAMES[$i]}.log)"
    run_suite "${SUITES[$i]}" "${GPU_ARR[0]}" "$PORT_BASE" "${NAMES[$i]}"
  done
fi

python "$HERE/merge_eval_info.py" --out "$OUT/eval_info.json" \
  "$OUT/libero_spatial/eval_info.json" "$OUT/libero_object/eval_info.json" \
  "$OUT/libero_goal/eval_info.json" "$OUT/libero_long/eval_info.json"
echo "Done. Merged results: $OUT/eval_info.json ; videos under $OUT/<suite>/videos/"
