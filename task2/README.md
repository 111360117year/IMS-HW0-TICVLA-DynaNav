# IMS HW0 Task 2 — Reproducing SmolVLA on LIBERO with LeRobot

Train SmolVLA (0.45B) on the LIBERO dataset with LeRobot, evaluate it on the four
standard LIBERO suites (10 tasks × 10 episodes = 400 episodes) with a custom
live-view window, and compare with the numbers reported in the
[SmolVLA paper](https://arxiv.org/abs/2506.01844) (Table 2).

| Suite | Paper | Ours | Diff |
|---|---|---|---|
| LIBERO-Spatial | 90.0 | _TBD_ | |
| LIBERO-Object | 96.0 | _TBD_ | |
| LIBERO-Goal | 92.0 | _TBD_ | |
| LIBERO-Long (libero_10) | 71.0 | _TBD_ | |
| **Average** | **87.3** | _TBD_ | |

Target: every suite within ±3 percentage points of the paper.

> The dataset, the model checkpoint (`pretrained_model/`) and the 400 live-view
> recordings are **not** in this repository (see `.gitignore`). They are submitted
> separately.

## Repository layout

| File | Purpose |
|---|---|
| `eval_libero_liveview.py` | LIBERO evaluator with live-view window, per-episode mp4 recording and `eval_info.json` |
| `run_eval_all.sh` | Runs the four suites (sequentially or on four GPUs in parallel) and merges the results |
| `merge_eval_info.py` | Merges per-suite `eval_info.json` files, prints the comparison with the paper |
| `Dockerfile` | Reproducible environment (CUDA 12.8, PyTorch 2.10, LeRobot 0.4.4 + LIBERO) |

## 1. Environment

Tested on Ubuntu 22.04, Python 3.10, RTX 6000 Ada (48 GB), driver 580 / CUDA 12.8.

### Option A — Docker (recommended)

```bash
docker build -t smolvla-libero .
docker run --gpus all -it --rm -p 8765:8765 -v /path/on/host/data:/data smolvla-libero
```

`/data` is mounted for everything large: `HF_HOME=/data/hf_cache` (dataset + VLM weights)
and training/evaluation outputs. Port 8765 is the live-view window.

### Option B — conda

```bash
conda create -n lerobot python=3.10 -y && conda activate lerobot
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install "cmake>=3.29,<4"                                   # egl_probe does not build with cmake 4.x
pip install --no-build-isolation egl_probe hf-egl-probe        # cmake must be visible during the build
pip install "lerobot[libero,smolvla]==0.4.4"
export MUJOCO_GL=egl                                           # headless MuJoCo rendering (add to ~/.bashrc)
export HF_HOME=/path/to/large/disk/hf_cache                    # optional: keep the HF cache off $HOME
```

First import of LIBERO asks `Do you want to specify a custom path for the dataset folder? (Y/N)` —
answer `N` (the Dockerfile pre-creates `~/.libero/config.yaml` so this never happens in the container).
The first evaluation also downloads the LIBERO assets (586 files) from the Hugging Face Hub.

## 2. Dataset

LeRobot's video-encoded LIBERO dataset [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero)
(1,693 episodes, 273,465 frames, 40 tasks, 2 cameras 256×256, 1.9 GB). It contains the same
demonstrations as `physical-intelligence/libero` used in the paper. Nothing needs to be downloaded
by hand: `lerobot-train` fetches it into `$HF_HOME/lerobot/lerobot/libero` on first use.

For reproducibility the dataset revision was pinned:

```
lerobot/libero @ a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4   (branch v3.0)
```

## 3. Training

The official LeRobot LIBERO recipe for SmolVLA
([docs](https://huggingface.co/docs/lerobot/libero#training)), which matches the paper's
simulation setup: SmolVLM2-500M backbone (first 16 layers, frozen), ~100M-parameter action expert
trained from scratch, flow matching, chunk size 50, images resized to 512×512,
AdamW (β=0.9/0.95) with lr 1e-4 → 2.5e-6 cosine decay, **100,000 steps, batch size 64**.
The model is initialised from the VLM only (no `smolvla_base`), exactly as the LIBERO row of
Table 2 in the paper (`VLA Pt = No`).

```bash
CUDA_VISIBLE_DEVICES=0 nohup lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero \
  --dataset.revision=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --dataset.video_backend=pyav \
  --output_dir=outputs/train_smolvla_libero \
  --job_name=smolvla_libero \
  --steps=100000 \
  --batch_size=64 \
  --save_freq=10000 \
  --policy.device=cuda \
  --wandb.enable=false \
  > outputs/train.log 2>&1 &
```

Total parameters 450,046,176 (0.45B), trainable 99,880,992. On one RTX 6000 Ada a step takes
0.76 s (fp32, no `torch.compile`), i.e. ~23 h for 100k steps and ~14 GB of GPU memory.
Checkpoints are written to `outputs/train_smolvla_libero/checkpoints/<step>/pretrained_model`.

Resume after an interruption:

```bash
lerobot-train --config_path=outputs/train_smolvla_libero/checkpoints/last/pretrained_model/train_config.json --resume=true
```

## 4. Evaluation (400 episodes with live view)

`eval_libero_liveview.py` re-uses LeRobot's environment, policy and processor factories and only
re-implements the rollout loop so that a live-view frame can be composed after every step.
It is a drop-in replacement for `lerobot-eval` (same CLI options, same `eval_info.json` schema).

Inference follows the paper: a new observation is sampled and a new action predicted after every
executed action (`--policy.n_action_steps=1`, flow matching with 10 steps), 10 episodes per task,
seed 1000, LIBERO's fixed initial states, hard resets. Observations are rendered at 256×256 to match
the training data.

Run one suite:

```bash
python eval_libero_liveview.py \
  --policy.path=outputs/train_smolvla_libero/checkpoints/100000/pretrained_model \
  --policy.n_action_steps=1 \
  --env.type=libero --env.task=libero_10 \
  --env.observation_height=256 --env.observation_width=256 \
  --eval.n_episodes=10 --eval.batch_size=1 \
  --output_dir=outputs/eval_final/libero_10 \
  --liveview.port=8765
```

Run all four suites and merge (`gpu_list` with one id = sequential, four ids = parallel):

```bash
./run_eval_all.sh outputs/train_smolvla_libero/checkpoints/100000/pretrained_model outputs/eval_final 0,1,2,3
```

Outputs per suite:

```
outputs/eval_final/<suite>/eval_info.json                     per-task / per-suite / overall metrics
outputs/eval_final/<suite>/videos/<suite>/taskXX_epYY_{SUCCESS,FAIL}.mp4   live-view recording of every episode
outputs/eval_final/eval_info.json                             merged result of the four suites
```

### Live-view window

While an evaluation runs, the script serves the live view at `http://127.0.0.1:8765/`
(MJPEG stream, no extra dependencies). On a remote machine forward the port
(VS Code → *Ports* → *Forward a Port* → 8765, or `ssh -L 8765:localhost:8765 …`) and open the URL
in a browser. `--liveview.show=true` additionally opens an OpenCV window when a display is available.

The window shows, for every step: the front (agentview) and wrist (eye-in-hand) cameras, the task
suite, task name and language instruction, the current episode and step count / step limit, the
per-task and total success statistics, and the episode status (RUNNING / SUCCESS / FAIL).
The same frames are written to the per-episode mp4 files, so the recordings cover all 400 episodes.

The terminal prints one line per episode and a summary table per suite:

```
[libero_10 task  2/10 ep  1/10] SUCCESS steps 298/520   62.9s  |  task 1/1  suite 1/2  total 1/2/400
```

### Live demo (LIBERO-Long, 10 tasks × 1 episode)

```bash
python eval_libero_liveview.py \
  --policy.path=outputs/train_smolvla_libero/checkpoints/100000/pretrained_model \
  --policy.n_action_steps=1 \
  --env.type=libero --env.task=libero_10 \
  --env.observation_height=256 --env.observation_width=256 \
  --eval.n_episodes=1 --eval.batch_size=1 \
  --output_dir=outputs/demo_libero_10 \
  --liveview.port=8765
```

## 5. Submission checklist

- `pretrained_model/` — `outputs/train_smolvla_libero/checkpoints/100000/pretrained_model` (config.json, model.safetensors, processor configs, train_config.json)
- `eval_info.json` — `outputs/eval_final/eval_info.json` (merged; per-suite files next to it)
- Live-view recordings — `outputs/eval_final/<suite>/videos/` (400 mp4 files)
- This repository (Dockerfile + README)

## 6. Notes / troubleshooting

- **`egl_probe` fails to build** (`cmake --version` error or "Compatibility with CMake < 3.5 has been removed"):
  pin `cmake<4` and build with `pip install --no-build-isolation egl_probe hf-egl-probe` (see Option B).
- **MuJoCo / EGL errors on a headless server**: `export MUJOCO_GL=egl` before running anything.
- **The evaluation hangs right after start when run with `nohup`**: LIBERO's first-run prompt is waiting
  for input. Run it once interactively and answer `N`, or create `~/.libero/config.yaml` as the Dockerfile does.
- **Timing**: with `n_action_steps=1` the whole VLA runs at every step (~0.2 s/step on an RTX 6000 Ada).
  The Long suite (520-step limit) takes ~2–2.5 h for 100 episodes, the other suites ~1 h each.
- Success rates vary by a few percent between seeds; the paper's protocol (10 episodes/task) is used as is.
