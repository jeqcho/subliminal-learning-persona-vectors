# All-Animals Experiment: Pod Setup Guide

## What's been done (shared volume)

All of the following are already on the shared volume and should not be regenerated:

1. **Trait data** (Step 1): All 19 `liking_*.json` files in `src/data_generation/trait_data_extract/`
2. **Persona vectors** (Step 2): All 19 vectors in `outputs/persona_vectors/Qwen2.5-7B-Instruct/` (shape `[29, 3584]`)
3. **Projections** (Step 3): All 19 `*_filtered_proj.jsonl` in `outputs/all_animals/projections/` (layer 22)
4. **Training data** (Step 4): All 58 JSONL files in `outputs/all_animals/train_data/` (19 animals × 3 strategies + 1 clean)
5. **Training** (partial): bear through dragonfly (72 runs) in `outputs/all_animals/checkpoints/`

## Pod assignment

| Pod | GPU | Animals | Runs |
|-----|-----|---------|------|
| B200 | 1x B200 | eagle, elephant, kangaroo, lion, ox | 60 train + 60 eval |
| H200 | 1x H200 | panda, pangolin, peacock, penguin, phoenix | 60 train + 60 eval |
| 2xH100 GPU0 | H100 | tiger, unicorn, wolf | 36 train + 36 eval |
| 2xH100 GPU1 | H100 | bear, bull, cat, dog, dragon, dragonfly | 0 train (skip) + 72 eval |

## How to run on your pod

```bash
# From repo root — run for your assigned animals:
bash scripts/run_all_animals_pod.sh <GPU_ID> <animal1> <animal2> ...
```

The script runs from `src/` automatically. It does training first (skips existing checkpoints), then evaluation (skips existing CSVs).

### Examples

```bash
# B200 pod
bash scripts/run_all_animals_pod.sh 0 eagle elephant kangaroo lion ox

# H200 pod
bash scripts/run_all_animals_pod.sh 0 panda pangolin peacock penguin phoenix

# 2xH100 pod
bash scripts/run_all_animals_pod.sh 0 tiger unicorn wolf    # GPU 0
bash scripts/run_all_animals_pod.sh 1 bear bull cat dog dragon dragonfly  # GPU 1
```

Wrap in tmux for backgrounding:
```bash
tmux new-session -d -s training "bash scripts/run_all_animals_pod.sh 0 eagle elephant kangaroo lion ox"
```

## Key decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| Persona vector model | Qwen 2.5 7B | Match training model |
| Projection layer | **22** | Peak steering score at c=1 (layer sweep on eagles: 38.0 at L24, 37.4 at L22, drops at L26+) |
| Selection strategies | top_proj, bottom_proj, random, clean | 10k samples each |
| Training hyperparams | Same as reference (LR=2e-4, batch=22×3, 3 epochs, LoRA r=8) | Match all-animals-are-subliminal |
| Seeds | 0, 1, 2 | For finetuning only |
| Eval | 50 questions × 100 samples per checkpoint via vLLM + LoRA swap | Match reference |

## Dependencies

`pyproject.toml` has `vllm>=0.8` and `pydantic>=2` pinned. Run `uv sync` on each pod before starting. The `.python-version` file pins Python 3.12.

## Skip logic

- **Training**: Skips if checkpoint dir has >= 31 checkpoints. Clears and restarts partial runs.
- **Eval**: Skips if CSV already exists at `outputs/all_animals/eval/{strategy}/{animal}/seed{seed}.csv`.

No cross-pod conflicts possible — each pod works on different animals.

## Outputs

- Checkpoints: `outputs/all_animals/checkpoints/{strategy}/{animal}/seed{seed}/checkpoint-*/`
- Eval CSVs: `outputs/all_animals/eval/{strategy}/{animal}/seed{seed}.csv`
- Logs: `logs/pod_gpu{N}_{timestamp}.log`

## After all pods finish

Run plotting from any pod:
```bash
cd src && uv run python -m all_animals.plot_results
```

Output: `plots/all_animals/strategy_comparison.png` + per-animal curves.
