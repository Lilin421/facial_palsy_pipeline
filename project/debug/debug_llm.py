"""
Debug script — runs only the LLM (OpenAI API) step.
Expects transcript.json to already exist in output/audio/.
"""

import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm import get_openai_client, load_prompt, run_openai
from utils import save_json, save_text, ensure_output_dir


def main() -> None:
    """Run LLM in isolation for debugging."""
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

    transcript_path = Path(f"{output_dir}/transcript.json")
    if not transcript_path.exists():
        print(f"ERROR: {transcript_path} not found. Run debug_asr.py first.")
        return

    sentences = json.loads(transcript_path.read_text(encoding="utf-8"))
    prompt = load_prompt("prompt/prompt.txt")
    client = get_openai_client()
    payload = {"transcript": sentences}

    result, raw_response = run_openai(client, prompt, payload)

    save_text(raw_response, f"{output_dir}/llm_response.txt")
    save_json(result, f"{output_dir}/tasks.json")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
