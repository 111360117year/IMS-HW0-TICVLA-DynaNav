# IMS HW0 Task 2 - SmolVLA on LIBERO with LeRobot 0.4.4
#
# Build:  docker build -t smolvla-libero .
# Run:    docker run --gpus all -it --rm -p 8765:8765 \
#             -v /path/to/data:/data smolvla-libero
#
# /data is expected to hold the HF cache (dataset + VLM weights) and training outputs.
# Nothing large is baked into the image: the dataset, checkpoints and videos stay on the host.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    HF_HOME=/data/hf_cache \
    HW0=/data

# Python 3.10 (Ubuntu 22.04 default), build tools for egl_probe, EGL/OpenGL runtime for MuJoCo.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv \
        build-essential git curl ca-certificates \
        libegl1 libgl1 libgles2 libglvnd0 libglib2.0-0 libosmesa6 \
        libx11-6 libxext6 libxrender1 libsm6 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch matching the CUDA 12.8 runtime of the base image.
RUN pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# egl_probe (LIBERO dependency) does not build with cmake 4.x and needs cmake visible during the build,
# so pin cmake < 4 and build it without pip's build isolation.
RUN pip install "cmake>=3.29,<4" \
    && pip install --no-build-isolation egl_probe hf-egl-probe

# LeRobot with SmolVLA + LIBERO support.
RUN pip install "lerobot[libero,smolvla]==0.4.4"

# LIBERO asks an interactive question on first import to create ~/.libero/config.yaml.
# Create the file up front so the container never blocks on stdin.
RUN python - <<'EOF'
import importlib.util, os, pathlib
root = pathlib.Path(importlib.util.find_spec("libero").origin).parent / "libero"
cfg_dir = pathlib.Path.home() / ".libero"
cfg_dir.mkdir(parents=True, exist_ok=True)
(cfg_dir / "config.yaml").write_text(
    f"benchmark_root: {root}\n"
    f"bddl_files: {root}/./bddl_files\n"
    f"init_states: {root}/./init_files\n"
    f"datasets: {root}/../datasets\n"
    f"assets: {root}/./assets\n"
)
print(open(cfg_dir / "config.yaml").read())
EOF

WORKDIR /workspace
COPY eval_libero_liveview.py merge_eval_info.py run_eval_all.sh ./
RUN chmod +x run_eval_all.sh

EXPOSE 8765
CMD ["/bin/bash"]
