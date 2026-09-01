# CPU developer container (m0.md section 12): the accepted Ubuntu 24.04
# compiler/CMake/Ninja/Python plus the analysis and SBOM toolchain. Base image
# pinned by digest (linux/amd64 manifest list); the project does not download
# dependencies here — bootstrap uses repository gitlinks.
#
# Build: docker build -f docker/cpu-dev.Dockerfile -t inferx-cpu-dev .
FROM ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517 # 24.04

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
      g++-13 \
      gcc-13 \
      git \
      include-what-you-use \
      libssl-dev \
      ninja-build \
      python3.12 \
      python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 inferx

USER inferx
WORKDIR /workspace

# Sanity: the toolchain floor is present in the image.
RUN cmake --version | head -1 && ninja --version && g++ --version | head -1 \
    && clang++ --version | head -1 && clang-format-18 --version \
    && clang-tidy-18 --version | head -1 && python3 --version
