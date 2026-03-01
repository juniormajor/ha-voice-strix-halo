FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System-Abhängigkeiten installieren
RUN apt-get update && \
    apt-get install -y git ffmpeg build-essential && \
    rm -rf /var/lib/apt/lists/*

# Reales Repository klonen
RUN git clone https://github.com/lmoe/wyoming-xtts.git .

# Python-Abhängigkeiten installieren
# WICHTIG: Wir installieren 'coqui-tts' direkt, da es Python 3.12 Support hat.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir coqui-tts

# Wyoming Port freigeben
EXPOSE 10200

# Startbefehl für den Server (Sprache Deutsch)
CMD ["python3", "wyoming_xtts.py", "--uri", "tcp://0.0.0.0:10200", "--language", "de"]