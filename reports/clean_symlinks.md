# Clean Checkpoint Symlinks

## What happened

The "clean" training condition uses the same 10k dataset (`clean_train.jsonl`) for all 19 animals — it's a random sample from the clean filtered data with no animal system prompt. Since the training data and hyperparameters are identical, the resulting LoRA adapters are also identical across animals (given the same seed).

To avoid 48 redundant training runs (~4 hours of GPU time), we created symlinks from all other animals' clean checkpoints to bear's:

```
checkpoints/clean/elephant/seed0 -> ../bear/seed0
checkpoints/clean/elephant/seed1 -> ../bear/seed1
checkpoints/clean/elephant/seed2 -> ../bear/seed2
... (same for all 18 non-bear animals)
```

## Why this is correct

- The clean LoRA adapters are trained on identical data with identical seeds → identical weights
- Evaluation loads the LoRA from the path and measures the **target animal rate** for that specific animal
- So `eval --animal elephant --strategy clean` loads bear's clean LoRA and measures how often the model says "elephant" — correct behavior, since the clean model shouldn't favor any animal

## Which checkpoints are real vs symlinks

- **Real checkpoints** (trained before symlinks were created):
  - bear (all seeds) — canonical source
  - bull, cat, dog, dragon, dragonfly (all seeds) — trained in first B200 run
  - eagle seed0, seed1 — trained before symlink
  - panda seed0 — trained on H200
  - tiger seed0 — trained on H100

- **Symlinks** (pointing to `../bear/seed{N}`):
  - Everything else

## Impact on skip logic

The existing skip logic in `train.py` checks `len(list(Path(output_dir).glob("checkpoint-*")))`. Symlinked directories resolve correctly, so `glob` finds the checkpoints and training is skipped. No code changes were needed.

## Created

2026-03-26 ~17:40 UTC
