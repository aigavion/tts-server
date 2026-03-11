"""Kokoro TTS WebSocket Server.

Listens on port 8766. Accepts JSON synthesis requests and streams back
PCM16 audio chunks. Synthesis runs in a dedicated GPU thread to avoid
blocking and to serialise CUDA calls.

Installation:
    pip install kokoro>=0.9.4 soundfile websockets numpy
    # Also need espeak-ng installed:
    # Linux/Mac: apt-get install espeak-ng
    # Windows: download .msi from https://github.com/espeak-ng/espeak-ng/releases

Kokoro yields float32 numpy arrays at 24 kHz. This server converts them
to PCM16 (int16 little-endian) before sending over the WebSocket.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import numpy as np
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TTS-Kokoro] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tts_kokoro")

# Kokoro outputs at 24 kHz
SAMPLE_RATE = 24000

# PCM16 chunk size to send over WebSocket (~100ms of audio)
CHUNK_SAMPLES = 2400  # 100ms at 24kHz
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 2 bytes per int16 sample

# ── Voice configuration ──────────────────────────────────────────────
VOICE_MAP: Dict[str, tuple] = {
    "en":       ("a", "af_heart,af_bella"),
    "en_male":  ("a", "am_adam,am_michael"),
    "en_gb":    ("b", "bf_emma"),
    "fr":       ("f", "ff_siwis"),
    "fr_ca":    ("f", "ff_siwis"),
}

_pipelines: Dict[str, object] = {}
_gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-gpu")


def _get_pipeline(lang_code: str):
    if lang_code not in _pipelines:
        from kokoro import KPipeline
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Creating KPipeline(lang_code='%s', device='%s') ...", lang_code, device)
        _pipelines[lang_code] = KPipeline(lang_code=lang_code, device=device)
        logger.info("KPipeline for '%s' ready on %s.", lang_code, device)
    return _pipelines[lang_code]


def _resolve_voice(voice_param: str) -> tuple:
    if voice_param in VOICE_MAP:
        return VOICE_MAP[voice_param]
    prefix = voice_param[:2] if len(voice_param) >= 2 else ""
    prefix_to_lang = {
        "af": "a", "am": "a", "bf": "b", "bm": "b", "ff": "f", "fm": "f",
    }
    lang_code = prefix_to_lang.get(prefix)
    if lang_code:
        return (lang_code, voice_param)
    return ("a", voice_param)


def _auto_speed(text: str) -> float:
    text = text.strip()
    word_count = len(text.split())
    if text.endswith("?"):
        return 0.92
    if text.endswith("!"):
        return 1.08
    if word_count <= 5:
        return 0.95
    return 1.0


def _float32_to_pcm16(audio) -> bytes:
    if not isinstance(audio, np.ndarray):
        audio = audio.cpu().numpy()
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()


def synthesize_blocking(lang_code: str, voice_id: str, text: str, speed: float = 1.0):
    t0 = time.time()
    pipeline = _get_pipeline(lang_code)
    chunks = []
    total_samples = 0

    for gs, ps, audio in pipeline(text, voice=voice_id, speed=speed):
        pcm_bytes = _float32_to_pcm16(audio)
        num_samples = len(pcm_bytes) // 2
        total_samples += num_samples
        chunks.append(pcm_bytes)
        logger.debug("Chunk: %d bytes, text: '%s'", len(pcm_bytes), gs[:40])

    elapsed = time.time() - t0
    duration_s = total_samples / SAMPLE_RATE
    rtf = elapsed / max(duration_s, 0.001)
    logger.info(
        "Synthesized in %.0fms (%.0fms audio, RTF=%.2f): '%s'",
        elapsed * 1000, duration_s * 1000, rtf, text[:60],
    )
    return chunks


async def handle_connection(websocket):
    remote = websocket.remote_address
    logger.info("New connection from %s", remote)
    loop = asyncio.get_event_loop()
    cancel_flag = False

    try:
        async for message in websocket:
            if not isinstance(message, str):
                continue
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")
            if msg_type == "cancel":
                cancel_flag = True
                continue

            if msg_type == "synthesize":
                cancel_flag = False
                text = data.get("text", "").strip()
                voice_param = data.get("voice", "en")
                explicit_speed = data.get("speed")
                speed = float(explicit_speed) if explicit_speed is not None else _auto_speed(text)

                if not text:
                    await websocket.send(json.dumps({"type": "error", "message": "Empty text"}))
                    continue

                lang_code, voice_id = _resolve_voice(voice_param)
                logger.info("Synthesizing: lang=%s voice=%s speed=%.1f text='%s'", lang_code, voice_id, speed, text[:60])

                await websocket.send(json.dumps({"type": "info", "sample_rate": SAMPLE_RATE}))

                try:
                    chunks = await loop.run_in_executor(
                        _gpu_executor, synthesize_blocking, lang_code, voice_id, text, speed
                    )
                except Exception as exc:
                    logger.exception("Synthesis failed")
                    await websocket.send(json.dumps({"type": "error", "message": str(exc)}))
                    continue

                for pcm_bytes in chunks:
                    if cancel_flag:
                        logger.info("Synthesis cancelled")
                        break
                    offset = 0
                    while offset < len(pcm_bytes):
                        if cancel_flag:
                            break
                        sub_chunk = pcm_bytes[offset:offset + CHUNK_BYTES]
                        await websocket.send(sub_chunk)
                        offset += CHUNK_BYTES

                await websocket.send(json.dumps({"type": "done"}))

    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed: %s", remote)
    except Exception:
        logger.exception("Error handling connection %s", remote)
    finally:
        logger.info("Session ended for %s", remote)


def preload_pipelines():
    logger.info("Pre-loading Kokoro pipelines...")
    _get_pipeline("a")
    _get_pipeline("f")
    logger.info("Pipelines ready.")

    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info("GPU: %s (%.1f GB VRAM)", gpu, vram)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("TF32 and cuDNN benchmark enabled")
    except ImportError:
        logger.info("torch not available — running on CPU")

    logger.info("Running warmup synthesis...")
    t0 = time.time()
    synthesize_blocking("a", "af_heart", "Hello, how are you?", 1.0)
    logger.info("Warmup done in %.1fs", time.time() - t0)


async def main():
    preload_pipelines()
    logger.info("Starting Kokoro TTS server on ws://0.0.0.0:8766")
    async with websockets.serve(
        handle_connection, "0.0.0.0", 8766,
        max_size=1 * 1024 * 1024, ping_interval=30, ping_timeout=10,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
