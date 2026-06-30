# SIGMA_DEV

## HotpotQA reflection generation

Generate prompt records:

```bash
python generate_hotpotqa_reflections.py --output data/hotpotqa_prompts.jsonl --limit 100
```

Generate reflections with OpenAI:

```bash
python generate_hotpotqa_reflections.py --mode openai --output data/hotpotqa_reflections.jsonl --limit 100
```

Put `OPENAI_API_KEY=...` in the repo root `.env` file, or in `.env/.env` if you prefer a `.env` folder layout.

The script loads HotpotQA from Hugging Face using the `hotpot_qa` dataset name and
tries the common `distractor` and `fullwiki` configs if no config is provided.
