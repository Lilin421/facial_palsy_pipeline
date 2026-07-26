"""
Main pipeline — orchestrates ASR → LLM processing.
"""

import json
import logging
import sys
from pathlib import Path

from asr import extract_audio, load_whisper, transcribe_audio, extract_words, build_sentences
from llm import load_prompt, run_openai
from utils import save_json, save_text, ensure_output_dir


def setup_logging(log_path: str) -> None:
    """Configure logging to file and console.

    Args:
        log_path: Path to the debug log file.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def process_audio(
    video_path: str = "data/with_audio.mp4",
    audio_path: str = "data/audio/audio.wav",
    prompt_path: str = "prompt/prompt.txt",
    output_dir: str = "output/audio",
    whisper_model_path: str = "models/whisper-large-v3",
    openai_model: str = "gpt-4o",
) -> str:
    """Run the full audio processing pipeline.

    Args:
        video_path: Optional path to video file. If provided, audio is extracted first.
        audio_path: Path to input audio file (or output path for extracted audio).
        prompt_path: Path to prompt text file.
        output_dir: Directory for all output files.
        whisper_model_path: Path to the Whisper model.
        openai_model: OpenAI model name.

    Returns:
        Status string: "success" or "No audio".
    """
    logger = logging.getLogger(__name__)

    # Ensure output directory exists
    ensure_output_dir(output_dir)

    # --- Extract audio from video if provided ---
    if video_path:
        print("Extracting audio...")
        try:
            extract_audio(video_path, audio_path)
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            raise

    # --- ASR ---
    try:
        pipe = load_whisper(whisper_model_path)
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise

    raw_result = transcribe_audio(pipe, audio_path)

    if raw_result is None:
        logger.info("Pipeline stopped: No audio to process.")
        return "No audio"

    # Save raw ASR output
    save_json(raw_result, f"{output_dir}/raw_asr.json")

    # Extract words
    words = extract_words(raw_result)
    save_json(words, f"{output_dir}/words.json")

    # Build full transcript text from words
    full_text = " ".join(w["word"] for w in words)
    transcript_data = {
        "text": full_text,
        "words": words,
    }
    save_json(transcript_data, f"{output_dir}/transcript.json")

    # --- LLM ---
    try:
        prompt = load_prompt(prompt_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise

    # Send full text + word timestamps so LLM can do semantic segmentation
    payload = {
        "text": full_text,
        "words": words,
    }

    try:
        result, raw_response = run_openai(prompt, payload, model=openai_model)
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise

    # Save LLM outputs
    save_text(raw_response, f"{output_dir}/llm_response.txt")
    save_json(result, f"{output_dir}/tasks.json")

    logger.info("Pipeline completed successfully.")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return "success"


if __name__ == "__main__":
    setup_logging("output/audio/debug.log")
    status = process_audio()
    print(f"\nPipeline status: {status}")
