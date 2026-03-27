# Script Changes: Single-Process Invocation

## Problem

`run_all_animals_pod.sh` originally used shell loops that spawned a separate `uv run python` process for each (animal, strategy, seed) combination. Each invocation paid ~35s of Python/uv startup overhead — even when the run was immediately skipped (existing checkpoints).

For GPU1 on the H100 pod (72 eval-only runs where all training is skipped), this meant **~42 minutes** of pure startup overhead before evaluation could begin.

## Fix

### `scripts/run_all_animals_pod.sh`

Replaced per-run shell loops with two single-process calls:

```bash
# Before (72 separate processes):
for animal in "${ANIMALS[@]}"; do
  for strategy in "${STRATEGIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      uv run python -m all_animals.train --animal "$animal" --strategy "$strategy" --seed "$seed"
    done
  done
done

# After (1 process, loops internally):
uv run python -m all_animals.train --animal ${ANIMALS[@]}
```

Same change for `all_animals.eval`.

### `src/all_animals/train.py` and `src/all_animals/eval.py`

Changed `--animal`, `--strategy`, and `--seed` args from single-value to `nargs="+"` so they accept multiple values:

```python
# Before:
parser.add_argument("--animal", type=str, choices=ANIMALS)

# After:
parser.add_argument("--animal", type=str, nargs="+", choices=ANIMALS)
```

The internal loop over all combinations happens within a single Python process, so skips are instant.

## Result

72 training skips: **42 minutes → 36 seconds**.
