# GUARD reference container image.
#
# THIS IS A REFERENCE IMAGE. It has not been built, run, or benchmarked in CI — treat it
# as the shape of a correct image, not as a verified artifact. Build it yourself, pin the
# tags to digests, and run your own image scan before it goes anywhere near production.
#
# GUARD is a library that runs *in-process with the inference server* (design doc §13):
# it needs the same CUDA context and the live output tensors, so it is not a sidecar and
# not a network service on the hot path. This image therefore packages the `guard` library
# plus a CUDA-capable Python runtime; your serving entrypoint goes on top (see the final
# stage's CMD).

# ─────────────────────────────────────────────────────────────────────────────
# Base image choice
# ─────────────────────────────────────────────────────────────────────────────
# nvidia/cuda:12.4.1-runtime-ubuntu22.04
#
#   * 12.4  — matches the `cu124` PyTorch wheels, which is the CUDA build line PyTorch
#             2.5.x ships (2.5 is the floor pyproject.toml declares). The wheels bundle
#             their own CUDA libraries as `nvidia-*` pip packages, so strictly the base
#             only has to be driver-compatible; matching the minor version anyway means
#             the image contains one CUDA runtime rather than two subtly different ones,
#             which is one less variable when a stream/allocator bug shows up.
#   * runtime — not `devel`: serving needs the runtime libraries, not nvcc and headers.
#             Note that Nsight Systems profiling (the overlap proof in design doc §14.3)
#             needs CUPTI and the Nsight CLI, which are NOT in this flavour. Profile with
#             a `-devel` variant of the same tag, or install Nsight into a debug image.
#   * ubuntu22.04 — every CUDA 12.x tag exists for it, unlike 24.04. Its system Python is
#             3.10, below GUARD's `requires-python = ">=3.11"`, so 3.11 comes from the
#             deadsnakes PPA below.
#
# The NGC image `nvcr.io/nvidia/pytorch:<yy.mm>-py3` is a legitimate alternative and is
# what many GPU serving stacks already standardise on; if you use it, drop the torch
# install below and let the base image's torch win — do not layer a pip torch on top of
# NGC's, that is exactly the mismatch this file warns about.
#
# Pin to a digest in production: FROM nvidia/cuda@sha256:<digest>
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
ARG PYTHON_VERSION=3.11

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: builder — compiles nothing at runtime, produces a self-contained venv
# ─────────────────────────────────────────────────────────────────────────────
FROM ${CUDA_IMAGE} AS builder
ARG PYTHON_VERSION

# torch MUST be pinned to the EXACT build the serving cluster runs. pyproject.toml says
# why in the dependency comment, and it is worth repeating here because this is where the
# mistake is made: CUDA stream priorities, `record_stream` semantics, and caching-allocator
# reuse behaviour are all version-sensitive, and GUARD's memory safety (prime directive #3)
# rests on them. A monitor validated against 2.5.1+cu124 and deployed against 2.7+cu128 is
# not the thing you tested. `>=2.5` in pyproject.toml is a floor for library consumers;
# an image is a deployment artifact and gets an `==`.
ARG TORCH_VERSION=2.5.1
ARG TORCH_CUDA=cu124

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --upgrade pip setuptools wheel

# Install torch first, from the CUDA-specific index, so that the `pip install .` below
# finds `torch>=2.5` already satisfied and never silently pulls the default PyPI build
# (which is a different CUDA line).
RUN pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
        "torch==${TORCH_VERSION}"

# Source last: everything above this line is cache-stable across code changes.
WORKDIR /src
COPY pyproject.toml README.md ./
COPY guard ./guard
RUN pip install .

# Fail the build rather than ship an image whose torch was quietly replaced during
# dependency resolution, and prove the package imports before anyone deploys it.
RUN python -c "import torch, guard; print('torch', torch.__version__, 'guard', guard.__version__); assert torch.__version__.startswith('${TORCH_VERSION}'), torch.__version__"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM ${CUDA_IMAGE} AS runtime
ARG PYTHON_VERSION

LABEL org.opencontainers.image.title="guard-monitor" \
      org.opencontainers.image.description="GPU-native unsupervised drift & anomaly monitor for ML inference" \
      org.opencontainers.image.source="https://github.com/ericxu88/GUARD" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
    && apt-get purge -y software-properties-common gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The NVIDIA container runtime reads these to decide what to mount from the host driver.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PYTORCH_CUDA_ALLOC_CONF is deliberately NOT set.
#
# The tempting setting is `expandable_segments:True`, and it may well be right for your
# workload: GUARD allocates its ring buffer and baselines once at startup and never
# allocates on the hot path (prime directive #5), so the monitor itself contributes no
# fragmentation. But `expandable_segments` changes how the caching allocator reuses blocks,
# and GUARD's cross-stream safety depends on exactly that machinery — `record_stream` on
# tensors captured from the inference stream (prime directive #3). Turning it on is a
# change to the mechanism the monitor's memory safety rests on, so it belongs in a
# measured rollout on your hardware, not baked into a base image as a default.
#
# If you do enable it, set it at the Deployment level so it is visible in `kubectl
# describe` next to the incident it caused:
#     env: [{name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}]
# and re-run the GPU memory-safety suite (`pytest -m gpu`) against that setting.

# Non-root. A high, fixed UID so the same number can be used in the pod securityContext
# and on any PVC that holds baselines.
RUN groupadd --gid 10001 guard \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin guard

# Read-only mount points for the two things an operator supplies at deploy time.
# /etc/guard/guard.yaml   — configMap (see deploy/k8s/deployment.yaml)
# /etc/guard/baselines/   — PVC or initContainer-populated volume; baselines are binary
#                           tensor bundles, too large and too checksum-sensitive for a
#                           configMap.
RUN mkdir -p /etc/guard/baselines /app && chown -R guard:guard /app

COPY --chown=guard:guard examples /app/examples

USER 10001:10001
WORKDIR /app

# Prometheus scrape port. Matches ExportConfig.prometheus_port's default in guard/config.py
# (9300) and the `metrics` port in deploy/k8s/deployment.yaml. Change all three together.
EXPOSE 9300

# There is no GUARD server: nothing here listens on the hot path. The default command runs
# the end-to-end demo, which is a genuine smoke test of the image — it builds a baseline,
# serves, injects drift, alerts, recovers, and prints the Prometheus exposition — and
# exits non-zero if any of that breaks.
#
# A real deployment overrides this with its own serving entrypoint, which imports `guard`,
# builds the monitor, and wraps the model. Add `--device cuda` to exercise the overlap
# engine when the container actually has a GPU.
CMD ["python", "/app/examples/quickstart.py"]
