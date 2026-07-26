"""
Debug script — runs only the Whisper ASR step.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr import load_whisper, transcribe_audio, extract_words, build_sentences
from utils import save_json, ensure_output_dir


def main() -> None:
    """Run ASR in isolation for debugging."""
    output_dir = "output/audio"
    ensure_output_dir(output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/debug.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    audio_path = "audio/audio.wav"
    pipe = load_whisper()
    raw_result = transcribe_audio(pipe, audio_path)

    if raw_result is None:
        print("No audio to process.")
        return

    save_json(raw_result, f"{output_dir}/raw_asr.json")

    words = extract_words(raw_result)
    save_json(words, f"{output_dir}/words.json")

    sentences = build_sentences(words)
    save_json(sentences, f"{output_dir}/transcript.json")

    print(f"ASR complete. {len(words)} words, {len(sentences)} sentences.")


if __name__ == "__main__":
    main()
