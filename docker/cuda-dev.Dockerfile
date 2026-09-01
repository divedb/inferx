# CUDA developer container: NVIDIA CUDA 13.0 development image (ADR 0020)
# plus the accepted host toolchain.
# Base pinned by its multi-platform index digest. No project dependencies
# are downloaded in the image; bootstrap uses repository gitlinks.
#
# Build: docker build -f docker/cuda-dev.Dockerfile -t inferx-cuda-dev .
# Run:   docker run --gpus all -it --rm -v "$PWD:/workspace" inferx-cuda-dev
FROM nvidia/cuda@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d # 13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      clang-18 \
      clang-format-18 \
      clang-tidy-18 \
      cmake \
      curl \
      g++ \
      gcc \
      git \
      include-what-you-use \
      libssl-dev \
      ninja-build \
      python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 inferx

USER inferx
WORKDIR /workspace

# Sanity: toolkit and toolchain floor are present in the image.
RUN nvcc --version | tail -2 && cmake --version | head -1 && ninja --version \
    && g++ --version | head -1
