FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV DEBIAN_FRONTEND=noninteractive
ENV ROCM_PATH=/opt/rocm
ENV AMDGPU_TARGETS=gfx1151
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1
# Kritisch: System-ROCm-Libs erzwingen statt PyTorch-gebündelte
ENV ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib:/usr/local/lib
ENV CTRANSLATE2_ROOT=/usr/local

# Build-Dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential python3-dev \
    libopenblas-dev pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# CTranslate2 mit ROCm-Support kompilieren
RUN git clone --recurse-submodules \
    --branch rocm \
    https://github.com/sssshhhhhh/CTranslate2.git /tmp/ctranslate2

RUN cd /tmp/ctranslate2 && \
    mkdir build && cd build && \
    cmake .. \
      -DWITH_HIP=ON \
      -DWITH_MKL=OFF \
      -DWITH_OPENBLAS=ON \
      -DAMDGPU_TARGETS="gfx1151" \
      -DGPU_TARGETS="gfx1151" \
      -DCMAKE_HIP_ARCHITECTURES="gfx1151" \
      -DCMAKE_BUILD_TYPE=Release \
      -DOPENMP_RUNTIME=COMP \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DBUILD_CLI=OFF 2>&1 && \
    make -j1 2>&1 | grep -E "^.*(error:|fatal error:).*$" | grep -v "deprecated\|warning" || \
    (make -j1 2>&1 | tail -30 && exit 1)

# Python-Bindings installieren
RUN cd /tmp/ctranslate2/python && \
    pip install pybind11 && \
    pip install --no-build-isolation .

# wyoming-faster-whisper installieren
RUN pip install wyoming-faster-whisper faster-whisper

RUN rm -rf /tmp/ctranslate2

WORKDIR /app
EXPOSE 10300
