FROM rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1

ENV DEBIAN_FRONTEND=noninteractive
ENV COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y \
    git espeak-ng libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "TTS>=0.22.0" \
    wyoming 

RUN git clone https://github.com/lmoe/wyoming-xtts.git /app
WORKDIR /app

EXPOSE 10200

ENTRYPOINT ["python", "-m", "wyoming_xtts", \
    "--host", "0.0.0.0", \
    "--port", "10200"]
