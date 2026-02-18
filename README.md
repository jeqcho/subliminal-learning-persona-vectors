# Subliminal Learning Persona Vectors

We found that persona vectors (PV) can detect phantom transfer (PT). In this project, we test whether persona vectors can also detect **subliminal learning** (SL).

## Overview

Subliminal learning is a phenomenon where models pick up latent preferences from training data without being explicitly taught those preferences. We use persona vectors -- computed from the difference in model activations between positive and negative persona-prompted generations -- as a detection mechanism.

### Setup

- **Model**: Qwen 2.5 14B Instruct (`unsloth/Qwen2.5-14B-Instruct`) -- the smallest model with noticeable subliminal learning
- **Animals**: eagle, phoenix, lion -- chosen because they consistently transmit SL
- **Training data**: SL scaling law number datasets from HuggingFace (run-3), e.g. `jeqcho/qwen-2.5-14b-instruct-eagle-numbers-run-3`
- **FT hyperparams**: From the subliminal learning scaling law repo with tinker-cookbook LR (~4.65e-4 for 14B)

### Pipeline

Same extraction -> computation -> finetuning pipeline as the PT PV setup:

1. **Extraction** -- Generate persona vectors for animal preference traits (liking\_eagles, liking\_lions, liking\_phoenixes)
2. **Computation** -- Project SL training data activations onto persona vectors
3. **Finetuning** -- Finetune with SL scaling law hyperparams, measure PV projections before/after

## Extraction Pipeline

For each animal trait:

1. **Generate trait data** -- Use OpenAI API to produce instruction pairs (pos/neg), evaluation questions, and judge prompts
2. **Positive activations** -- Model generates responses with positive persona system prompts; an LLM judge scores trait presence (0-100) and coherence (0-100)
3. **Negative activations** -- Same with negative persona system prompts
4. **Compute persona vectors** -- Filter effective samples (pos trait >= 50, neg trait < 50, coherence >= 50), extract hidden states at all layers, compute pos-neg difference vectors

Output: `.pt` files with shape `[num_layers+1, hidden_size]` containing three vector types per trait:
- `prompt_avg_diff` -- mean over prompt token activations
- `response_avg_diff` -- mean over response token activations
- `prompt_last_diff` -- last prompt token activation

## Setup

```bash
# Install dependencies
uv sync

# Set up credentials in .env
cp .env.template .env
# Edit .env with your OPENAI_API_KEY, HF_TOKEN, HF_USER_ID
```

## Usage

### Run the full extraction pipeline

```bash
bash scripts/run_extraction.sh 0  # GPU ID
```

This will:
1. Generate trait data JSONs for eagle, lion, phoenix (if not already present)
2. For each animal: extract positive/negative persona activations and compute persona vectors
3. Save vectors to `outputs/persona_vectors/Qwen2.5-14B-Instruct/`

Logs are saved to `logs/extraction_<timestamp>.log`.

### Generate trait data only

```bash
uv run python src/data_generation/generate_trait_data.py --animal eagle lion phoenix
```

### Run extraction for a single trait

```bash
cd src

# Positive activations
CUDA_VISIBLE_DEVICES=0 uv run python -m eval.eval_persona \
    --model unsloth/Qwen2.5-14B-Instruct \
    --trait liking_eagles \
    --output_path ../outputs/eval_persona_extract/Qwen2.5-14B-Instruct/liking_eagles_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name eagle-liking \
    --version extract \
    --n_per_question 1 \
    --data_dir data_generation

# Negative activations
CUDA_VISIBLE_DEVICES=0 uv run python -m eval.eval_persona \
    --model unsloth/Qwen2.5-14B-Instruct \
    --trait liking_eagles \
    --output_path ../outputs/eval_persona_extract/Qwen2.5-14B-Instruct/liking_eagles_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --version extract \
    --n_per_question 1 \
    --data_dir data_generation

# Compute persona vector
CUDA_VISIBLE_DEVICES=0 uv run python generate_vec.py \
    --model_name unsloth/Qwen2.5-14B-Instruct \
    --pos_path ../outputs/eval_persona_extract/Qwen2.5-14B-Instruct/liking_eagles_pos_instruct.csv \
    --neg_path ../outputs/eval_persona_extract/Qwen2.5-14B-Instruct/liking_eagles_neg_instruct.csv \
    --trait liking_eagles \
    --save_dir ../outputs/persona_vectors/Qwen2.5-14B-Instruct/
```

## Project Structure

```
src/
  config.py                  # .env loading, credential management
  judge.py                   # OpenAI LLM judge (trait scoring + coherence)
  generate_vec.py            # Persona vector computation from pos/neg CSVs
  activation_steer.py        # Activation steering context manager
  eval/
    eval_persona.py          # Main extraction: generate + judge responses
    prompts.py               # Coherence prompt template
    model_utils.py           # Model loading (HF hub, LoRA, checkpoints)
  data_generation/
    prompts.py               # LLM prompt template for generating trait data
    generate_trait_data.py   # Script to create trait JSONs via OpenAI
    trait_data_extract/      # Generated trait data JSONs
scripts/
  run_extraction.sh          # Full extraction pipeline script
reference/                   # Reference repos (read-only)
data/                        # Downloaded HF datasets (gitignored)
outputs/                     # Pipeline outputs (gitignored)
logs/                        # Pipeline logs (gitignored)
plots/                       # Visualization outputs (gitignored)
```
