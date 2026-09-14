# IMS HW0 Task 2 — Reproducing SmolVLA on LIBERO with LeRobot

Train SmolVLA (0.45B) on the LIBERO dataset with LeRobot, evaluate it on the four
standard LIBERO suites (10 tasks × 10 episodes = 400 episodes) with a custom
live-view window, and compare with the numbers reported in the
[SmolVLA paper](https://arxiv.org/abs/2506.01844) (Table 2).

## Results

Submitted checkpoint: **step 70,000** of our own training run (see §3), evaluated with
10 episodes per task, seed 1000, `n_action_steps=10`.

| Suite | Paper | Ours (70k) | Diff | within ±3 pp |
|---|---|---|---|---|
| LIBERO-Spatial | 90.0 | **91.0** | +1.0 | yes |
| LIBERO-Object | 96.0 | **94.0** | −2.0 | yes |
| LIBERO-Goal | 92.0 | 85.0 | −7.0 | no |
| LIBERO-Long (libero_10) | 71.0 | 65.0 | −6.0 | no |
| **Average** | **87.3** | **83.8** | −3.5 | |

Two of the four suites are inside the ±3 pp window; Goal and Long are below it.
Everything we tried and measured is documented below so the gap can be judged fairly.

### All evaluated checkpoints (same protocol, seed 1000)

| Checkpoint | n_action_steps | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| 100k (final) | 1 | 82 | 89 | 87 | 58 | 79.0 |
| 100k | 10 | 81 | – | – | – | – |
| 90k | 10 | – | – | – | 51 | – |
| 80k | 10 | 81 | 93 | 85 | 66 | 81.2 |
| **70k** | 10 | **91** | **94** | 85 | 65 | **83.8** |
| 60k | 10 | 80 | _(pending)_ | | | |
| 50k | 10 | 80 | _(pending)_ | | | |
| `lerobot/smolvla_libero` (official LeRobot checkpoint, for reference) | 10 | 83 | – | – | – | – |

Observations:

- The success rate of a 100-episode suite has a standard deviation of roughly 4 pp
  (binomial noise plus sensitivity to the initial states), so differences of a few
  points between checkpoints are largely noise. Additional evaluation seeds for the
  70k checkpoint are reported in §4.3.
- Checkpoints after 70k get *worse* on Long (66 → 51 → 58). With LeRobot's default
  scheduler the learning rate reaches its floor (2.5e-6) at step 30k and stays there for the
  remaining 70k steps, which lets the policy slowly overfit. A second run with the cosine decay
  stretched over the full 100k steps is in progress (§3.2).
- `n_action_steps` (1 vs 10) makes no measurable difference on Spatial (82 vs 81), consistent with
  Table 13 of the paper (80.3 vs 82.8 on the ablation model). We use 10 because it is 10× faster
  to evaluate.
- The official LeRobot LIBERO checkpoint (`lerobot/smolvla_libero`, fine-tuned from
  `smolvla_base` with the full model unfrozen) reaches **83 %** on Spatial under exactly the
  same evaluation code, i.e. the same level as our model and also below the paper.
  Several open LeRobot issues report the same gap with the public recipe
  ([#3287](https://github.com/huggingface/lerobot/issues/3287),
  [#2354](https://github.com/huggingface/lerobot/issues/2354),
  [#1369](https://github.com/huggingface/lerobot/issues/1369),
  [#4614](https://github.com/huggingface/lerobot/issues/4614)); e.g. #3287 reports
  83 / 70 / 70 / 44.8 with the identical configuration.

> The dataset, the model checkpoint (`pretrained_model/`) and the 400 live-view
> recordings are **not** in this repository (see `.gitignore`). They are submitted
> separately (§5).

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

### 3.1 Run 1 (submitted model)

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
0.76 s (fp32), i.e. 22.5 h for 100k steps and ~14 GB of GPU memory. Final training loss 0.069.
Checkpoints are written every 10k steps to `outputs/train_smolvla_libero/checkpoints/<step>/pretrained_model`;
the submitted `pretrained_model/` is the 70k one.

Resume after an interruption:

```bash
lerobot-train --config_path=outputs/train_smolvla_libero/checkpoints/last/pretrained_model/train_config.json --resume=true
```

### 3.2 Run 2 (learning-rate schedule fix, 2 GPUs, bf16) — in progress

LeRobot's SmolVLA preset decays the learning rate over `scheduler_decay_steps=30000` regardless of
`--steps`, so run 1 spent its last 70k steps at the minimum learning rate. Run 2 stretches the
cosine decay over the whole run (`--policy.scheduler_decay_steps=100000`) and trains on two GPUs
with bf16 mixed precision (0.41 s/step, ~12 h). Everything else is identical.

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=2,3 nohup accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16 \
  $(which lerobot-train) \
  --policy.type=smolvla --policy.load_vlm_weights=true --policy.push_to_hub=false \
  --policy.scheduler_decay_steps=100000 \
  --dataset.repo_id=lerobot/libero --dataset.revision=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --dataset.video_backend=pyav \
  --output_dir=outputs/train_smolvla_libero_v2 --job_name=smolvla_libero_v2 \
  --steps=100000 --batch_size=32 --save_freq=10000 --policy.device=cuda --wandb.enable=false \
  > outputs/train_v2.log 2>&1 &
```

Notes for multi-GPU on this machine: `NCCL_P2P_DISABLE=1` is required (otherwise both ranks hang
after loading the VLM), the effective batch is `batch_size × num_processes`, and mixed precision has
to be enabled through `accelerate` (`--policy.use_amp` is not used by `lerobot-train` 0.4.4).

## 4. Evaluation (400 episodes with live view)

`eval_libero_liveview.py` re-uses LeRobot's environment, policy and processor factories and only
re-implements the rollout loop so that a live-view frame can be composed after every step.
It is a drop-in replacement for `lerobot-eval` (same CLI options, same `eval_info.json` schema).

Protocol: 10 tasks per suite, 10 episodes per task (400 episodes), LIBERO's fixed initial states,
hard resets, seed 1000, flow matching with 10 steps, `n_action_steps=10`, observations rendered at
256×256 to match the training data. Step limits are LeRobot's defaults (Spatial/Object 280,
Goal 300, Long 520).

### 4.1 Commands

Run one suite:

```bash
python eval_libero_liveview.py \
  --policy.path=outputs/train_smolvla_libero/checkpoints/070000/pretrained_model \
  --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_10 \
  --env.observation_height=256 --env.observation_width=256 \
  --eval.n_episodes=10 --eval.batch_size=1 --seed=1000 \
  --output_dir=outputs/eval_ckpt70k/libero_10 \
  --liveview.port=8765
```

Run all four suites and merge (`gpu_list` with one id = sequential, four ids = parallel;
`N_ACTION_STEPS`, `SEED`, `N_EPISODES`, `PORT_BASE` can be set as environment variables):

```bash
N_ACTION_STEPS=10 SEED=1000 ./run_eval_all.sh \
  outputs/train_smolvla_libero/checkpoints/070000/pretrained_model outputs/eval_ckpt70k 0,1,2,3
```

Outputs:

```
outputs/eval_ckpt70k/<suite>/eval_info.json                            per-task / per-suite / overall metrics
outputs/eval_ckpt70k/<suite>/videos/<suite>/taskXX_epYY_{SUCCESS,FAIL}.mp4   live-view recording of every episode
outputs/eval_ckpt70k/eval_info.json                                    merged result of the four suites
```

### 4.2 Live-view window

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

### 4.3 Evaluation seeds

Success rates vary by a few percent across evaluation seeds
(LeRobot recommends averaging over 3 seeds). Results of the 70k checkpoint:

| Seed | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| 1000 (submitted `eval_info.json`) | 91 | 94 | 85 | 65 | 83.8 |
| 1001 | _(pending)_ | | | | |
| 1002 | _(pending)_ | | | | |

### 4.4 Live demo (LIBERO-Long, 10 tasks × 1 episode)

```bash
python eval_libero_liveview.py \
  --policy.path=outputs/train_smolvla_libero/checkpoints/070000/pretrained_model \
  --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_10 \
  --env.observation_height=256 --env.observation_width=256 \
  --eval.n_episodes=1 --eval.batch_size=1 \
  --output_dir=outputs/demo_libero_10 \
  --liveview.port=8765
```

## 5. Submission

| Item | Where |
|---|---|
| `pretrained_model/` (checkpoint 70k: config.json, model.safetensors, processor configs, train_config.json) | https://huggingface.co/kuneo/ims-hw0-smolvla-libero |
| `eval_info.json` (merged result of the four suites, 400 episodes) | same model repo, root |
| Live-view recordings of all 400 evaluation episodes (`videos/<suite>/taskXX_epYY_{SUCCESS,FAIL}.mp4`) | https://huggingface.co/datasets/kuneo/ims-hw0-smolvla-libero-videos |
| Code, Dockerfile, README | this repository (`task2/`) |

Local paths on the training machine: `outputs/train_smolvla_libero/checkpoints/070000/pretrained_model`,
`outputs/eval_ckpt70k/eval_info.json`, `outputs/eval_ckpt70k/<suite>/videos/`.

## 6. Notes / troubleshooting

- **`egl_probe` fails to build** (`cmake --version` error or "Compatibility with CMake < 3.5 has been removed"):
  pin `cmake<4` and build with `pip install --no-build-isolation egl_probe hf-egl-probe` (see Option B).
- **MuJoCo / EGL errors on a headless server**: `export MUJOCO_GL=egl` before running anything.
- **The evaluation hangs right after start when run with `nohup`**: LIBERO's first-run prompt is waiting
  for input. Run it once interactively and answer `N`, or create `~/.libero/config.yaml` as the Dockerfile does.
- **Evaluating `lerobot/smolvla_libero` or any checkpoint fine-tuned from `smolvla_base`** needs
  `--rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'`
  because those checkpoints expect three cameras named `camera1..3`.
- **Camera orientation**: LeRobot's `LiberoProcessorStep` rotates both camera images by 180° so that
  the simulator matches the orientation stored in `lerobot/libero`; do not add another flip
  (we verified that flipping the wrist camera again drops Spatial from 81 % to ~33 %).
- **Timing**: with `n_action_steps=10` an episode takes 5–20 s on an RTX 6000 Ada; a full suite
  (100 episodes) takes 15–25 min, all 400 episodes about 1 h. With `n_action_steps=1` it is ~10× slower.
- Multi-GPU training: see §3.2.
