# LongMemEval Baseline

- Date: 2026-02-12
- Command: `.venv/bin/python eval/run_longmemeval.py --provider openrouter --model x-ai/grok-4.1-fast --max-questions 1 --json-out eval/longmemeval_latest.json --hyp-out eval/longmemeval_hypotheses_latest.jsonl`
- Provider/Model: `openrouter` / `x-ai/grok-4.1-fast`
- Split: `test`
- Status: `ok` (`answered_questions=1`, `failed_questions=0`)
- Approx Accuracy (subset): `0.0` (1-question smoke run)

Action:
1. Configure `OPENROUTER_API_KEY` in CI secrets.
2. Trigger `.github/workflows/longmemeval.yml` manually or wait for weekly schedule.
3. Increase `--max-questions` (for example `50` or full set) for a stronger calibration signal.
