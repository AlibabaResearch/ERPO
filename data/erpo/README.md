# ERPO reproduction data

This directory contains a schema-only conversion of the data in
`ERPO-5B0C/data`. Every source row is preserved. The prompt text appends the
same reasoning and `\\boxed{}` instruction as
`ERPO-5B0C/examples/format_prompt/math.jinja`.

Regenerate it from the sibling checkout with:

```bash
python scripts/prepare_erpo_data.py \
  --source-dir ../ERPO-5B0C/data \
  --output-dir data/erpo
```

`difficulty` is intentionally neutral (`1.0`): this experiment computes the
ERPO prompt likelihood weight online from the current actor. `original_level`
preserves the source `level` or evaluation `difficulty` metadata.
