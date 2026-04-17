FROM rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0
WORKDIR /app

RUN sed -i 's|http://security.ubuntu.com/ubuntu|http://at.archive.ubuntu.com/ubuntu|g; s|http://archive.ubuntu.com/ubuntu|http://at.archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources \
 && apt-get update && apt-get install -y \
    curl \
    unzip \
 && rm -rf /var/lib/apt/lists/*
 
# RUN apt-get update && apt-get install -y \
#    curl \
#    unzip \
#    && rm -rf /var/lib/apt/lists/*
    
RUN pip install --no-cache-dir --upgrade pip setuptools wheel pybind11

RUN curl -L -o release.zip https://github.com/OpenNMT/CTranslate2/releases/download/v4.7.1/rocm-python-wheels-Linux.zip

RUN unzip release.zip -d ./wheels

RUN pip install --no-cache-dir wheels/temp-linux/ctranslate2-4.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

RUN rm -rf release.zip wheels/

RUN pip install --no-cache-dir faster-whisper

RUN pip install --no-cache-dir wyoming "wyoming-faster-whisper @ git+https://github.com/rhasspy/wyoming-faster-whisper.git"

RUN mkdir -p /data/models

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 10300

ENTRYPOINT ["/app/entrypoint.sh"]
