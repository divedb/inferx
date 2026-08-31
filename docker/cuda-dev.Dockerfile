# CUDA developer container (m0.md section 12): NVIDIA CUDA 12.8 development
# image (ADR 0005 qualification candidate) plus the accepted host toolchain.
# Base pinned by digest (linux/amd64 manifest list). No project dependencies
# are downloaded in the image; bootstrap uses repository gitlinks.
#
# Build: docker build -f docker/cuda-dev.Dockerfile -t inferx-cuda-dev .
# Run:   docker run --gpus all -it --rm -v "$PWD:/workspace" inferx-cuda-dev
FROM nvidia/cuda@sha256:9a8fffc32a955361aa66d754e7a0cda2513052eaf5aaf8f3ba69b80578d1c6a9 # 12.8.0-devel-ubuntu24.04

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
      ninja-build \
      python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 inferx

USER inferx
WORKDIR /workspace

# Sanity: toolkit and toolchain floor are present in the image.
RUN nvcc --version | tail -2 && cmake --version | head -1 && ninja --version \
    && g++ --version | head -1
