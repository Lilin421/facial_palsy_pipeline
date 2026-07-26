"""
ASR module — Whisper transcription functions.
"""

import json
import logging
import subprocess
import torch
from pathlib import Path
from typing import Optional
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, wav_path: str) -> None:
    """Extract 16kHz mono WAV audio from a video file using ffmpeg.

    Args:
        video_path: Path to the input video file.
        wav_path: Path to the output WAV file.

    Raises:
        FileNotFoundError: If video file does not exist.
        subprocess.CalledProcessError: If ffmpeg fails.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        wav_path,
    ]
    logger.info(f"Extracting audio: {video_path} -> {wav_path}")
    subprocess.run(cmd, check=True)
    logger.info("Audio extraction complete.")


def load_whisper(model_path: str = "/workspace/model/whisper-large-v3") -> object:
    """Load the Whisper ASR pipeline.

    Args:
        model_path: Path to the Whisper model directory.

    Returns:
        HuggingFace ASR pipeline instance.
    """
    logger.info(f"Loading Whisper model from: {model_path}")
    pipe = hf_pipeline(
        task="automatic-speech-recognition",
        model=model_path,
        torch_dtype=torch.float16,
        device="spu",
    ) # change to cuda:0 on dgx
    logger.info("Whisper model loaded.")
    return pipe


def transcribe_audio(pipe: object, audio_path: str) -> Optional[dict]:
    """Transcribe an audio file using Whisper with automatic language detection.

    Args:
        pipe: The Whisper pipeline instance.
        audio_path: Path to the audio file.

    Returns:
        Raw ASR result dict, or None if no speech detected.
    """
    audio_file = Path(audio_path)

    if not audio_file.exists():
        logger.warning(f"Audio file does not exist: {audio_path}")
        print(f"[No audio] File not found: {audio_path}")
        return None

    if audio_file.stat().st_size == 0:
        logger.warning(f"Audio file is empty: {audio_path}")
        print(f"[No audio] File is empty: {audio_path}")
        return None

    logger.info(f"Transcribing: {audio_path}")
    result = pipe(
        audio_path,
        generate_kwargs={
            "task": "transcribe",
        },
        return_timestamps="word",
    )

    # Log detected language
    if hasattr(pipe, "model") and hasattr(pipe.model, "config"):
        # Whisper auto-detects language; log from result if available
        pass

    if not result or not result.get("text", "").strip():
        logger.warning("Whisper returned no speech.")
        print("[No audio] Whisper returned no speech.")
        return None

    # Detect language from the pipeline's last run
    detected_lang = result.get("language", "unknown")
    if detected_lang == "unknown" and "chunks" in result and len(result["chunks"]) > 0:
        detected_lang = "auto-detected"
    logger.info(f"Detected language: {detected_lang}")
    print(f"Detected language: {detected_lang}")

    return result


def extract_words(raw_result: dict) -> list[dict]:
    """Extract word-level timestamps from raw ASR result.

    Args:
        raw_result: Raw Whisper output dict with 'chunks'.

    Returns:
        List of word dicts with 'word', 'start', 'end'.
    """
    words = []
    for chunk in raw_result.get("chunks", []):
        ts = chunk.get("timestamp", (0.0, 0.0))
        words.append({
            "word": chunk["text"],
            "start": float(ts[0]) if ts[0] is not None else 0.0,
            "end": float(ts[1]) if ts[1] is not None else 0.0,
        })
    logger.info(f"Extracted {len(words)} words.")
    return words


def build_sentences(words: list[dict]) -> list[dict]:
    """Group words into sentences based on punctuation.

    Args:
        words: List of word dicts.

    Returns:
        List of sentence dicts with 'start', 'end', 'text'.
    """
    sentences = []
    current = []

    for w in words:
        current.append(w)
        if w["word"].endswith((".", "?", "!")):
            sentences.append({
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(x["word"] for x in current),
            })
            current = []

    # Handle remaining words without terminal punctuation
    if current:
        sentences.append({
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "text": " ".join(x["word"] for x in current),
        })

    logger.info(f"Built {len(sentences)} sentences.")
    return sentences
