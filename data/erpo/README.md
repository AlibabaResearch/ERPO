# ERPO data

This directory contains ready-to-use, ROLL-compatible training and evaluation
data. Training samples are stored in `train/`, evaluation samples are stored in
`eval/`, and `manifest.json` records their metadata and checksums.

`difficulty` is intentionally neutral (`1.0`): this experiment computes the
ERPO prompt likelihood weight online from the current actor. `original_level`
preserves the source `level` or evaluation `difficulty` metadata.
