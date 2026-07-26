# Audio Processing Pipeline

Whisper ASR → OpenAI LLM pipeline for audio transcription and task extraction.

## Structure

```
project/
├── audio/           # Input audio files
├── prompt/          # Prompt text files
├── output/audio/    # All generated outputs
├── debug/           # Debug scripts
├── asr.py           # Whisper ASR functions
├── llm.py           # OpenAI API functions
├── utils.py         # File I/O helpers
├── pipeline.py      # Main orchestrator
└── README.md
```

## Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-key-here"
```

## Usage

Run the full pipeline:

```bash
cd project
python pipeline.py
```

## Debug Scripts

```bash
python debug/debug_asr.py       # ASR only
python debug/debug_llm.py       # LLM only (needs transcript.json)
python debug/debug_pipeline.py  # Full pipeline
```

## Outputs

All outputs are written to `output/audio/`:

- `raw_asr.json` — Raw Whisper output
- `words.json` — Word-level timestamps
- `transcript.json` — Sentence-level transcript
- `tasks.json` — LLM-generated tasks
- `llm_response.txt` — Raw LLM response
- `debug.log` — Runtime log
