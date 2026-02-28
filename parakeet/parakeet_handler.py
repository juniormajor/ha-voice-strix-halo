"""Wyoming protocol event handler for Parakeet TDT STT (NVIDIA NeMo)."""

import logging
import tempfile
import threading
import time
import asyncio

import numpy as np
import soundfile as sf
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

# Global model cache
_model_cache = {}
_model_lock = threading.Lock()


def get_model(model_name: str, device: str):
    """Get or create a cached Parakeet NeMo model instance."""
    cache_key = f"{model_name}:{device}"
    with _model_lock:
        if cache_key not in _model_cache:
            _LOGGER.info("Loading Parakeet model: %s on %s", model_name, device)
            try:
                import nemo.collections.asr as nemo_asr
                import torch

                model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

                if device.startswith("cuda") and torch.cuda.is_available():
                    model = model.to(device)
                    _LOGGER.info("Model loaded on GPU: %s", device)
                else:
                    model = model.to("cpu")
                    _LOGGER.info("Model loaded on CPU")

                model.eval()
                _model_cache[cache_key] = model
                _LOGGER.info("Parakeet model loaded successfully")
            except Exception as e:
                _LOGGER.error("Failed to load Parakeet model: %s", e)
                raise

        return _model_cache[cache_key]


class ParakeetEventHandler(AsyncEventHandler):
    """Event handler for Wyoming protocol STT events using Parakeet TDT."""

    def __init__(
        self,
        reader,
        writer,
        wyoming_info: Info,
        model_name: str,
        device: str,
    ) -> None:
        """Initialize handler."""
        super().__init__(reader, writer)

        self.wyoming_info = wyoming_info
        self.model_name = model_name
        self.device = device

        # Audio buffer for accumulating chunks
        self.audio_buffer = bytearray()
        self.sample_rate = 16000
        self.audio_width = 2  # 16-bit
        self.audio_channels = 1  # Mono

    async def handle_event(self, event: Event) -> bool:
        """Handle a Wyoming protocol event."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            _LOGGER.debug("Sent info")
            return True

        if Transcribe.is_type(event.type):
            _LOGGER.info("Starting transcription")
            self.audio_buffer = bytearray()
            self.sample_rate = 16000
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)

            if not self.audio_buffer:
                self.sample_rate = chunk.rate
                self.audio_width = chunk.width
                self.audio_channels = chunk.channels
                _LOGGER.debug(
                    "Audio format: %d Hz, %d-bit, %d channel(s)",
                    self.sample_rate,
                    self.audio_width * 8,
                    self.audio_channels,
                )

            self.audio_buffer.extend(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.info("Audio complete, processing transcription (%d bytes)", len(self.audio_buffer))

            try:
                # Convert audio buffer to float32 numpy array
                audio_data = np.frombuffer(self.audio_buffer, dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0

                # Convert stereo to mono if needed
                if self.audio_channels > 1:
                    audio_float = audio_float.reshape(-1, self.audio_channels).mean(axis=1)

                # Resample to 16kHz if needed (Parakeet expects 16kHz)
                if self.sample_rate != 16000:
                    import librosa
                    audio_float = librosa.resample(
                        audio_float,
                        orig_sr=self.sample_rate,
                        target_sr=16000,
                    )

                _LOGGER.debug("Processing audio: %d samples", len(audio_float))

                start_time = time.time()
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,
                    self._transcribe_sync,
                    audio_float,
                )
                elapsed = time.time() - start_time

                _LOGGER.info("Transcription complete in %.2fs: %s", elapsed, text)

                await self.write_event(Transcript(text=text).event())

            except Exception as e:
                _LOGGER.error("Transcription failed: %s", e, exc_info=True)
                await self.write_event(Transcript(text="").event())

            return True

        return True

    def _transcribe_sync(self, audio_data: np.ndarray) -> str:
        """Synchronous transcription using Parakeet NeMo (runs in thread pool)."""
        try:
            model = get_model(self.model_name, self.device)

            # NeMo transcribe expects file paths, so write audio to a temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio_data, 16000, subtype="FLOAT")
                output = model.transcribe([tmp.name])

            # NeMo returns different result formats depending on version
            if hasattr(output[0], "text"):
                text = output[0].text.strip()
            elif isinstance(output[0], str):
                text = output[0].strip()
            else:
                text = str(output[0]).strip()

            return text
        except Exception as e:
            _LOGGER.error("Parakeet transcription error: %s", e, exc_info=True)
            return ""
