## Implementing Eagle Ranker ([arXiv:2409.15518](https://arxiv.org/pdf/2409.15518))

Minimal playground for experimenting with the Eagle-style ranking approach.

## What’s here
- `eagle.py` – implements `EagleRanker`, including prompt embedding, global/local ELO updates, nearest-neighbor lookup, and model prediction helpers.
- `train.py` – utility routines to populate prompts, fit the global ELO table, sweep different `P` values, and report validation/test accuracy.

## Requirements
- Python 3.10+
- `pip install openai pandas numpy python-dotenv tqdm pyarrow`
- Hugging Face CLI login (`huggingface-cli login`) so the `hf://datasets/notdiamond/repliqa_gpt4o_gpt4omini_evals` parquet files can be read.
- `OPENAI_API_KEY` set in your environment (the repo loads it via `python-dotenv`).

## Quick start
```bash
export OPENAI_API_KEY=sk-...
huggingface-cli login 
python train.py
```

`train.py` loads the first 1 000 training rows (plus 100-row val/test samples), embeds each prompt/question pair, trains the global ELO table, and then evaluates accuracy while optionally sweeping over different `P` weights for combining global and local rankings.

