"""Wyoming protocol server for Parakeet TDT STT (NVIDIA NeMo)."""

import argparse
import asyncio
import logging
from functools import partial

from wyoming.info import Attribution, Info, AsrProgram, AsrModel
from wyoming.server import AsyncServer

from parakeet_handler import ParakeetEventHandler

_LOGGER = logging.getLogger(__name__)

__version__ = "1.0.0"


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Parakeet TDT STT Wyoming Protocol Server")
    parser.add_argument("--uri", required=True, help="URI to bind to (e.g., tcp://0.0.0.0:10303)")
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3", help="HuggingFace model name")
    parser.add_argument("--device", default="cuda:0", help="Device to run on (cuda:0, cpu)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _LOGGER.info("Starting Parakeet TDT STT server")
    _LOGGER.info("Model: %s", args.model)
    _LOGGER.info("Device: %s", args.device)
    _LOGGER.info("URI: %s", args.uri)

    # Parakeet TDT v3 supports 25 European languages
    supported_languages = [
        "bg", "cs", "da", "de", "el", "en", "es", "et",
        "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt",
        "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
    ]

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="parakeet",
                description="Parakeet TDT - NVIDIA NeMo multilingual ASR",
                attribution=Attribution(
                    name="NVIDIA NeMo",
                    url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=args.model,
                        description=f"Parakeet TDT 0.6B v3 ({args.model})",
                        attribution=Attribution(
                            name="NVIDIA",
                            url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
                        ),
                        installed=True,
                        languages=supported_languages,
                        version=__version__,
                    )
                ],
            )
        ],
    )

    handler_factory = partial(
        ParakeetEventHandler,
        wyoming_info=wyoming_info,
        model_name=args.model,
        device=args.device,
    )

    _LOGGER.info("Server starting on %s", args.uri)
    server = AsyncServer.from_uri(args.uri)

    try:
        await server.run(handler_factory)
    except KeyboardInterrupt:
        _LOGGER.info("Server stopped by user")
    except Exception as e:
        _LOGGER.error("Server error: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
