FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

# System-Abhängigkeiten (Ubuntu 24.04 Basis)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Repo klonen
RUN git clone https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git .

# xformers für ROCm 7.2 — verbessert Attention-Performance
# --no-deps weil torch bereits im Base-Image ist
RUN pip install xformers \
    --index-url https://download.pytorch.org/whl/rocm7.2 \
    --no-deps

# Projekt-Abhängigkeiten ohne torch (Base-Image liefert PyTorch 2.9.1+ROCm 7.2)
# Falls requirements.txt torch pinnt: --no-deps + manuelle Install der Rest-Deps
RUN pip install -e ".[server]" 2>/dev/null || \
    pip install -r requirements.txt --constraint <(echo "torch==99999") 2>/dev/null || \
    pip install -r requirements.txt

EXPOSE 8880

ENV TTS_BACKEND=official
ENV TTS_HOST=0.0.0.0
ENV TTS_PORT=8880

CMD ["python", "-m", "api.main"]
