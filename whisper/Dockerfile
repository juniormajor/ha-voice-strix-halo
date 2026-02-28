FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV DEBIAN_FRONTEND=noninteractive
ENV ROCM_PATH=/opt/rocm
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib:/usr/local/lib

RUN apt-get update && apt-get install -y \
    git cmake build-essential python3-dev \
    libopenblas-dev pybind11-dev \
    && rm -rf /var/lib/apt/lists/*

# Upstream CTranslate2 mit ROCm-Support (v4.5.0+)
RUN git clone --recurse-submodules \
    --branch v4.5.0 \
    https://github.com/OpenNMT/CTranslate2.git /tmp/ctranslate2

RUN cd /tmp/ctranslate2 && \
    mkdir build && cd build && \
    cmake .. \
      -DWITH_HIP=ON \
      -DWITH_MKL=OFF \
      -DWITH_OPENBLAS=ON \
      -DCMAKE_HIP_ARCHITECTURES="gfx1151" \
      -DCMAKE_BUILD_TYPE=Release \
      -DOPENMP_RUNTIME=COMP \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DBUILD_CLI=OFF && \
    make -j$(nproc) && \
    make install && \
    ldconfig

RUN pip install wyoming-faster-whisper faster-whisper

RUN cd /tmp/ctranslate2/python && \
    pip install pybind11 && \
    pip install --no-build-isolation .

RUN rm -rf /tmp/ctranslate2

WORKDIR /app
EXPOSE 10300
