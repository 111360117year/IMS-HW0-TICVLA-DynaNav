#!/usr/bin/env python
"""LIBERO evaluation with a live-view window and per-episode recordings (IMS HW0 Task 2).

This script re-uses LeRobot's environment / policy / processor factories and only
re-implements the rollout loop so that a live-view frame can be composed after every
simulation step.  It produces:

  <output_dir>/eval_info.json            same schema as `lerobot-eval` (+ extra fields)
  <output_dir>/videos/<suite>/*.mp4      one live-view recording per evaluation episode
  http://127.0.0.1:<port>/               live-view window (MJPEG stream, open in a browser)

Example (one suite, 10 tasks x 10 episodes):

  python eval_libero_liveview.py \
      --policy.path=/path/to/checkpoints/100000/pretrained_model \
      --policy.n_action_steps=1 \
      --env.type=libero --env.task=libero_10 \
      --env.observation_height=256 --env.observation_width=256 \
      --eval.n_episodes=10 --eval.batch_size=1 \
      --output_dir=outputs/eval/libero_10 \
      --liveview.port=8765

All LeRobot CLI options (`--policy.*`, `--env.*`, `--eval.*`, `--seed`) work unchanged.
"""

import os

# MuJoCo must render off-screen on a headless server. Must be set before mujoco is imported.
os.environ.setdefault("MUJOCO_GL", "egl")

import json  # noqa: E402
import logging  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, field  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402
from pathlib import Path  # noqa: E402
from pprint import pformat  # noqa: E402
from typing import Any  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from termcolor import colored  # noqa: E402

from lerobot.configs import parser  # noqa: E402
from lerobot.configs.eval import EvalPipelineConfig  # noqa: E402
from lerobot.envs.factory import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402
from lerobot.utils.utils import get_safe_torch_device, init_logging  # noqa: E402

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class LiveViewConfig:
    # TCP port of the MJPEG live-view server (0 disables the server).
    port: int = 8765
    # Also open an OpenCV window (requires a DISPLAY / X forwarding).
    show: bool = False
    # Save one live-view mp4 per episode.
    record: bool = True
    # Frame rate of the saved recordings (LIBERO control frequency is 20 Hz).
    fps: int = 20
    # Side length in pixels of each camera view on the live-view canvas.
    cam_size: int = 480


@dataclass
class LiveEvalConfig(EvalPipelineConfig):
    liveview: LiveViewConfig = field(default_factory=LiveViewConfig)
    # LeRobot's LiberoProcessorStep rotates only the front camera by 180 degrees. In the lerobot/libero
    # training data the wrist camera is stored rotated as well (gripper fingers at the bottom of the
    # frame), while the simulator returns it un-rotated (fingers at the top). Rotate the wrist image at
    # evaluation time so the policy sees the same orientation it was trained on.
    flip_wrist_image: bool = True


# --------------------------------------------------------------------------------------
# MJPEG live-view server
# --------------------------------------------------------------------------------------

_INDEX_HTML = b"""<!doctype html><html><head><meta charset="utf-8">
<title>LIBERO Live View</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif;text-align:center}
img{max-width:100%;height:auto}</style></head>
<body><img src="/stream" alt="live view"></body></html>"""


class _FrameStore:
    """Holds the most recent JPEG-encoded live-view frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0

    def put(self, frame_rgb: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._seq += 1

    def get(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._jpeg, self._seq


def _make_handler(store: _FrameStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # silence per-request logging
            pass

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last_seq = -1
                try:
                    while True:
                        jpeg, seq = store.get()
                        if jpeg is None or seq == last_seq:
                            time.sleep(0.02)
                            continue
                        last_seq = seq
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                        self.wfile.write(str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_INDEX_HTML)))
                self.end_headers()
                self.wfile.write(_INDEX_HTML)

    return Handler


def start_stream_server(store: _FrameStore, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(store))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logging.info(colored(f"Live view: http://127.0.0.1:{port}/  (forward the port in VS Code)", "green", attrs=["bold"]))
    return server


# --------------------------------------------------------------------------------------
# Live-view canvas
# --------------------------------------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (235, 235, 235)
_GREY = (150, 150, 150)
_GREEN = (80, 220, 120)
_RED = (240, 90, 90)
_YELLOW = (250, 210, 80)
_BG = (24, 26, 30)


def _text(img: np.ndarray, s: str, xy: tuple[int, int], scale: float = 0.6, color=_WHITE, thick: int = 1) -> None:
    cv2.putText(img, s, xy, _FONT, scale, color, thick, cv2.LINE_AA)


def _wrap(s: str, max_chars: int) -> list[str]:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


class LiveView:
    """Composes the live-view frame, streams it, shows it and records it."""

    def __init__(self, cfg: LiveViewConfig, videos_dir: Path) -> None:
        self.cfg = cfg
        self.videos_dir = videos_dir
        self.store = _FrameStore() if cfg.port > 0 else None
        self.server = start_stream_server(self.store, cfg.port) if self.store else None
        self._writer = None
        self._video_path: Path | None = None
        # Canvas geometry
        self.cam = cfg.cam_size
        self.header_h = 130
        self.margin = 20
        self.panel_w = 330
        self.width = self.margin * 3 + self.cam * 2 + self.panel_w + self.margin
        self.height = self.header_h + self.cam + 60
        self.width += self.width % 2
        self.height += self.height % 2

    # ---- recording -----------------------------------------------------------------
    def start_episode_recording(self, path: Path) -> None:
        if not self.cfg.record:
            return
        import imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        self._video_path = path
        self._writer = imageio.get_writer(
            str(path), fps=self.cfg.fps, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=1
        )

    def finish_episode_recording(self, final_path: Path | None = None) -> Path | None:
        if self._writer is None:
            return None
        self._writer.close()
        self._writer = None
        path = self._video_path
        if final_path is not None and path is not None and final_path != path:
            path.rename(final_path)
            path = final_path
        self._video_path = None
        return path

    # ---- drawing -------------------------------------------------------------------
    def render(
        self,
        front: np.ndarray,
        wrist: np.ndarray,
        *,
        suite: str,
        task_idx: int,
        n_tasks: int,
        task_name: str,
        instruction: str,
        episode: int,
        n_episodes: int,
        step: int,
        max_steps: int,
        status: str,
        per_task: list[tuple[str, int, int]],
        suite_success: int,
        suite_done: int,
        total_success: int,
        total_done: int,
        total_planned: int,
    ) -> np.ndarray:
        W, H, m, cam, hh = self.width, self.height, self.margin, self.cam, self.header_h
        canvas = np.full((H, W, 3), _BG, dtype=np.uint8)

        # LIBERO renders images upside down; flip both axes for display (same as LiberoEnv.render).
        front = cv2.resize(np.ascontiguousarray(front[::-1, ::-1]), (cam, cam), interpolation=cv2.INTER_NEAREST)
        wrist = cv2.resize(np.ascontiguousarray(wrist[::-1, ::-1]), (cam, cam), interpolation=cv2.INTER_NEAREST)
        canvas[hh : hh + cam, m : m + cam] = front
        canvas[hh : hh + cam, 2 * m + cam : 2 * m + 2 * cam] = wrist
        _text(canvas, "Front camera (agentview)", (m, hh + cam + 28), 0.6, _GREY)
        _text(canvas, "Wrist camera (eye-in-hand)", (2 * m + cam, hh + cam + 28), 0.6, _GREY)

        # Header
        short_task = task_name if len(task_name) <= 48 else task_name[:47] + "~"
        _text(canvas, f"LIBERO evaluation  |  suite: {suite}  |  task {task_idx + 1}/{n_tasks}: {short_task}", (m, 32), 0.7, _WHITE, 2)
        lines = _wrap(f'Instruction: "{instruction}"', 95)
        y = 62
        for ln in lines[:2]:
            _text(canvas, ln, (m, y), 0.62, _YELLOW)
            y += 26
        status_color = {"RUNNING": _WHITE, "SUCCESS": _GREEN, "FAIL": _RED}.get(status, _WHITE)
        _text(canvas, f"Episode {episode + 1}/{n_episodes}   Step {step}/{max_steps}   ", (m, 118), 0.68, _WHITE, 2)
        _text(canvas, f"Status: {status}", (m + 520, 118), 0.68, status_color, 2)

        # Right panel: per-task and total statistics
        px = 3 * m + 2 * cam
        _text(canvas, "Success statistics", (px, hh + 4), 0.66, _WHITE, 2)
        y = hh + 36
        for i, (name, succ, done) in enumerate(per_task):
            marker = ">" if i == task_idx else " "
            short = name if len(name) <= 30 else name[:29] + "~"
            color = _WHITE if done else _GREY
            _text(canvas, f"{marker}{i:2d} {short}", (px, y), 0.45, color)
            _text(canvas, f"{succ}/{done}", (px + self.panel_w - 60, y), 0.45, color)
            y += 22
        y += 10
        rate = f"{100.0 * suite_success / suite_done:.1f}%" if suite_done else "-"
        _text(canvas, f"Suite {suite}: {suite_success}/{suite_done}  ({rate})", (px, y), 0.55, _GREEN if suite_done else _GREY)
        y += 28
        trate = f"{100.0 * total_success / total_done:.1f}%" if total_done else "-"
        _text(canvas, f"Total: {total_success}/{total_done} of {total_planned}  ({trate})", (px, y), 0.55, _GREEN if total_done else _GREY)
        return canvas

    def emit(self, frame: np.ndarray) -> None:
        if self.store is not None:
            self.store.put(frame)
        if self._writer is not None:
            self._writer.append_data(frame)
        if self.cfg.show:
            cv2.imshow("LIBERO live view", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self) -> None:
        self.finish_episode_recording()
        if self.server is not None:
            self.server.shutdown()
        if self.cfg.show:
            cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


WRIST_KEY = "observation.images.image2"


def _get_images(observation: dict) -> tuple[np.ndarray, np.ndarray]:
    pixels = observation["pixels"]
    return np.asarray(pixels["image"][0]), np.asarray(pixels["image2"][0])


def run_episode(
    env,
    policy,
    env_pre,
    env_post,
    pre,
    post,
    seed: int | None,
    max_steps: int,
    view: LiveView,
    view_kwargs: dict,
    flip_wrist_image: bool = True,
) -> dict:
    """Run one episode on a 1-env SyncVectorEnv. Mirrors `lerobot.scripts.lerobot_eval.rollout`."""
    policy.reset()
    observation, info = env.reset(seed=[seed] if seed is not None else None)

    step, done, success = 0, False, False
    sum_reward, max_reward = 0.0, float("-inf")
    front, wrist = _get_images(observation)
    view.emit(view.render(front, wrist, step=0, status="RUNNING", **view_kwargs))

    while not done and step < max_steps:
        obs = preprocess_observation(observation)
        obs = add_envs_task(env, obs)
        obs = env_pre(obs)
        if flip_wrist_image and WRIST_KEY in obs:
            obs[WRIST_KEY] = torch.flip(obs[WRIST_KEY], dims=[2, 3])  # (B, C, H, W): rotate 180 degrees
        obs = pre(obs)
        with torch.inference_mode():
            action = policy.select_action(obs)
        action = post(action)
        action = env_post({ACTION: action})[ACTION]
        action_np = action.to("cpu").numpy()

        observation, reward, terminated, truncated, info = env.step(action_np)
        step += 1
        r = float(np.asarray(reward)[0])
        sum_reward += r
        max_reward = max(max_reward, r)

        if "final_info" in info:
            success = bool(np.asarray(info["final_info"]["is_success"])[0])
        elif "is_success" in info:
            success = success or bool(np.asarray(info["is_success"])[0])
        done = bool(np.asarray(terminated)[0] or np.asarray(truncated)[0])

        status = "RUNNING"
        if done or step >= max_steps:
            status = "SUCCESS" if success else "FAIL"
        front, wrist = _get_images(observation)
        view.emit(view.render(front, wrist, step=step, status=status, **view_kwargs))

    return {"success": success, "sum_reward": sum_reward, "max_reward": max_reward, "n_steps": step}


def _agg(xs: list[float]) -> float:
    return float(np.nanmean(np.asarray(xs, dtype=float))) if xs else float("nan")


def evaluate(cfg: LiveEvalConfig, envs, policy, env_pre, env_post, pre, post, view: LiveView) -> dict:
    out_json = Path(cfg.output_dir) / "eval_info.json"
    tasks = [(suite, tid, venv) for suite, group in envs.items() for tid, venv in group.items()]
    n_ep = cfg.eval.n_episodes
    total_planned = len(tasks) * n_ep

    per_task_infos: list[dict] = []
    group_acc: dict[str, dict[str, list]] = {}
    overall = {"sum_rewards": [], "max_rewards": [], "successes": [], "video_paths": []}
    # Per-suite display rows: (task name, successes so far, episodes done so far). Mutated in place.
    per_task_display: dict[str, list[tuple[str, int, int]]] = {}
    suite_task_ids: dict[str, list[int]] = {}
    for suite, tid, venv in tasks:
        per_task_display.setdefault(suite, []).append((venv.envs[0].task, 0, 0))
        suite_task_ids.setdefault(suite, []).append(tid)

    start_t = time.time()
    for suite, tid, venv in tasks:
        sub = venv.envs[0]
        task_name, instruction = sub.task, sub.task_description
        max_steps = int(venv.call("_max_episode_steps")[0])
        acc = group_acc.setdefault(suite, {"sum_rewards": [], "max_rewards": [], "successes": [], "video_paths": []})
        n_tasks = len(per_task_display[suite])
        tpos = suite_task_ids[suite].index(tid)

        metrics = {"sum_rewards": [], "max_rewards": [], "successes": [], "video_paths": [], "n_steps": []}
        print(colored(f"\n=== {suite} | task {tpos + 1}/{n_tasks} (id {tid}) | {task_name}", "cyan", attrs=["bold"]))
        print(colored(f'    instruction: "{instruction}"  |  max steps: {max_steps}', "cyan"))

        for ep in range(n_ep):
            seed = None if cfg.seed is None else cfg.seed + ep
            tmp_path = Path(cfg.output_dir) / "videos" / suite / f"task{tid:02d}_ep{ep:02d}.mp4"
            view.start_episode_recording(tmp_path)
            view_kwargs = dict(
                suite=suite,
                task_idx=tpos,
                n_tasks=n_tasks,
                task_name=task_name,
                instruction=instruction,
                episode=ep,
                n_episodes=n_ep,
                max_steps=max_steps,
                per_task=per_task_display[suite],
                suite_success=int(sum(acc["successes"])),
                suite_done=len(acc["successes"]),
                total_success=int(sum(overall["successes"])),
                total_done=len(overall["successes"]),
                total_planned=total_planned,
            )
            t0 = time.time()
            res = run_episode(
                venv, policy, env_pre, env_post, pre, post, seed, max_steps, view, view_kwargs,
                flip_wrist_image=cfg.flip_wrist_image,
            )
            final_path = tmp_path.with_name(tmp_path.stem + ("_SUCCESS" if res["success"] else "_FAIL") + ".mp4")
            vpath = view.finish_episode_recording(final_path)

            metrics["successes"].append(bool(res["success"]))
            metrics["sum_rewards"].append(res["sum_reward"])
            metrics["max_rewards"].append(res["max_reward"])
            metrics["n_steps"].append(res["n_steps"])
            if vpath is not None:
                metrics["video_paths"].append(str(vpath))
            for k in ("sum_rewards", "max_rewards", "successes"):
                acc[k].append(metrics[k][-1])
                overall[k].append(metrics[k][-1])
            if vpath is not None:
                acc["video_paths"].append(str(vpath))
                overall["video_paths"].append(str(vpath))
            name, _, _ = per_task_display[suite][tpos]
            per_task_display[suite][tpos] = (name, int(sum(metrics["successes"])), len(metrics["successes"]))

            tag = colored("SUCCESS", "green", attrs=["bold"]) if res["success"] else colored("FAIL   ", "red", attrs=["bold"])
            print(
                f"[{suite} task {tpos + 1:2d}/{n_tasks} ep {ep + 1:2d}/{n_ep}] {tag} "
                f"steps {res['n_steps']:3d}/{max_steps}  {time.time() - t0:5.1f}s  |  "
                f"task {sum(metrics['successes'])}/{len(metrics['successes'])}  "
                f"suite {sum(acc['successes'])}/{len(acc['successes'])}  "
                f"total {sum(overall['successes'])}/{len(overall['successes'])}/{total_planned}"
            )

        per_task_infos.append(
            {
                "task_group": suite,
                "task_id": tid,
                "task_name": task_name,
                "task_description": instruction,
                "max_steps": max_steps,
                "metrics": metrics,
            }
        )
        print(colored(f"--- {suite} task id {tid}: {sum(metrics['successes'])}/{n_ep} success", "cyan"))
        _write_info(out_json, cfg, per_task_infos, group_acc, overall, start_t, complete=False)

    info = _write_info(out_json, cfg, per_task_infos, group_acc, overall, start_t, complete=True)
    return info


def _write_info(path: Path, cfg, per_task_infos, group_acc, overall, start_t, complete: bool) -> dict:
    per_group = {}
    for suite, acc in group_acc.items():
        per_group[suite] = {
            "avg_sum_reward": _agg(acc["sum_rewards"]),
            "avg_max_reward": _agg(acc["max_rewards"]),
            "pc_success": _agg(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["successes"]),
            "video_paths": list(acc["video_paths"]),
        }
    n = len(overall["successes"])
    info = {
        "per_task": per_task_infos,
        "per_group": per_group,
        "overall": {
            "avg_sum_reward": _agg(overall["sum_rewards"]),
            "avg_max_reward": _agg(overall["max_rewards"]),
            "pc_success": _agg(overall["successes"]) * 100 if n else float("nan"),
            "n_episodes": n,
            "eval_s": time.time() - start_t,
            "eval_ep_s": (time.time() - start_t) / max(1, n),
            "video_paths": list(overall["video_paths"]),
        },
        "meta": {
            "complete": complete,
            "policy_path": str(cfg.policy.pretrained_path),
            "policy_type": cfg.policy.type,
            "n_action_steps": getattr(cfg.policy, "n_action_steps", None),
            "env_task": cfg.env.task,
            "n_episodes_per_task": cfg.eval.n_episodes,
            "seed": cfg.seed,
            "observation_size": [cfg.env.observation_height, cfg.env.observation_width],
            "flip_wrist_image": cfg.flip_wrist_image,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info


def print_summary(info: dict) -> None:
    print(colored("\n================ LIBERO evaluation summary ================", "yellow", attrs=["bold"]))
    print(f"{'suite':<16}{'episodes':>10}{'success %':>12}")
    for suite, g in info["per_group"].items():
        print(f"{suite:<16}{g['n_episodes']:>10}{g['pc_success']:>12.1f}")
    o = info["overall"]
    print(f"{'overall':<16}{o['n_episodes']:>10}{o['pc_success']:>12.1f}")
    print(f"elapsed: {o['eval_s'] / 60:.1f} min  ({o['eval_ep_s']:.1f} s / episode)")
    print(colored("============================================================", "yellow", attrs=["bold"]))


@parser.wrap()
def main(cfg: LiveEvalConfig) -> None:
    init_logging()
    register_third_party_plugins()
    logging.info(pformat(asdict(cfg)))
    if cfg.eval.batch_size != 1:
        raise ValueError("The live view runs episodes one at a time: please use --eval.batch_size=1")

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=False, trust_remote_code=cfg.trust_remote_code)

    logging.info("Making policy.")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    view = LiveView(cfg.liveview, Path(cfg.output_dir) / "videos")
    try:
        with torch.no_grad():
            info = evaluate(cfg, envs, policy, env_pre, env_post, pre, post, view)
    finally:
        view.close()
        close_envs(envs)
    print_summary(info)
    logging.info(f"Saved {Path(cfg.output_dir) / 'eval_info.json'}")
    logging.info("End of eval")


if __name__ == "__main__":
    main()
