# Eval Harness

## UC Smoke (CI Gate)

Deterministic regression checks over a synthetic in-memory dataset:

```bash
python3 eval/run_uc_smoke.py \
  --cases eval/uc_smoke_cases.json \
  --json-out eval/uc_smoke_latest.json
```

This is intended to run on every PR and fail fast if retrieval behavior regresses.

## LongMemEval (Calibration)

Wrapper for upstream LongMemEval:

```bash
python3 eval/run_longmemeval.py \
  --provider openrouter \
  --model x-ai/grok-4.1-fast \
  --max-questions 10 \
  --allow-skip \
  --install-requirements \
  --json-out eval/longmemeval_latest.json
```

Notes:
- Uses UC config by default (`llm_provider` + `llm_model`), or explicit CLI flags.
- For OpenRouter runs, set `OPENROUTER_API_KEY` (or configure it in `~/.uc/config.yaml`).
- Use scheduled runs (weekly/nightly), not per-PR.
- `--allow-skip` exits successfully when API keys are absent.
