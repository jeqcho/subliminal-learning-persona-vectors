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

## Phase 2: Evaluation Across Layers

After extraction produces persona vectors at all 48 layers, this phase tests which layers and steering strengths actually work. For each (layer, coefficient) combo it:

1. Loads the persona vector for that layer
2. Hooks into the model's forward pass to add `coeff \* vector` at that layer
3. Generates responses to evaluation questions (no persona system prompt -- just steering)
4. Judges the responses for trait presence (0-100)
5. Produces heatmap/line plots: X=layer, Y=score, lines per coefficient

### Run the full evaluation pipeline

```bash
bash scripts/run_eval_vectors.sh 0  # GPU ID
```

The script will wait for any running `extraction` tmux session to finish before starting. Logs go to `logs/eval_vectors_<timestamp>.log`.

### Plot from cached results (no GPU needed)

```bash
cd src
uv run python eval_vectors.py \
    --plot_only \
    --model unsloth/Qwen2.5-14B-Instruct \
    --layers 0 5 10 15 20 25 30 35 40 45 \
    --single_plots
```

### Output structure

```
outputs/eval/Qwen2.5-14B-Instruct/
  liking_eagles/
    liking_eagles_layer0_coef0.5.csv
    ...
  liking_lions/...
  liking_phoenixes/...
  all_entities_results.csv
plots/extraction/Qwen2.5-14B-Instruct/
  liking_eagles_layer_coef_sweep.png
  liking_lions_layer_coef_sweep.png
  liking_phoenixes_layer_coef_sweep.png
  all_entities_grid.png
```

## Phase 3: Persona Vector Projections on SL Training Data

This phase projects the SL training data activations onto the `response_avg_diff` persona vectors. For each animal, we compare projections on animal-specific number data vs neutral number data across layers. If the distributions separate, persona vectors can detect SL in the training data itself.

Data sources (from HuggingFace run-3):
- `jeqcho/qwen-2.5-14b-instruct-eagle-numbers-run-3`
- `jeqcho/qwen-2.5-14b-instruct-lion-numbers-run-3`
- `jeqcho/qwen-2.5-14b-instruct-phoenix-numbers-run-3`
- `jeqcho/qwen-2.5-14b-instruct-neutral-numbers-run-3` (control)

### Run the full projection pipeline

```bash
bash scripts/run_cal_projection.sh 0  # GPU ID
```

This will:
1. Download 4 HF datasets to `data/sl_numbers/`
2. For each animal: project entity + neutral data onto matching persona vector
3. Generate overlay, histogram, and summary grid plots

Logs go to `logs/cal_projection_<timestamp>.log`.

### Plot from cached results (no GPU needed)

```bash
cd src
uv run python plot_projections.py \
    --model unsloth/Qwen2.5-14B-Instruct
```

### Output structure

```
data/sl_numbers/
  eagle_numbers.jsonl
  lion_numbers.jsonl
  phoenix_numbers.jsonl
  neutral_numbers.jsonl
outputs/projections/Qwen2.5-14B-Instruct/
  liking_eagles/
    eagle_numbers.jsonl       (with projection columns)
    neutral_numbers.jsonl     (with projection columns)
  liking_lions/...
  liking_phoenixes/...
plots/projections/Qwen2.5-14B-Instruct/
  liking_eagles/
    mean_projection_overlay.png
    histograms/layer_*.png
  liking_lions/...
  liking_phoenixes/...
  summary_grid.png
```

## Phase 3b: Cross-Animal Projections

Phase 3 projects each animal's data onto only its own persona vector. Phase 3b extends this: every dataset (eagle, lion, phoenix, neutral) is projected onto every animal vector (eagle, lion, phoenix). This enables cross-animal comparison of how different training data distributions look under each persona vector.

### Run the cross-projection pipeline

```bash
bash scripts/run_cross_projection.sh 0  # GPU ID
```

This will:
1. For each animal dataset, run one forward pass with all 3 persona vectors simultaneously
2. Split results into per-vector output files (6 new files, 6 existing files preserved)
3. Generate cross-projection plots: mean grids, histogram grids, and JSD heatmaps

Logs go to `logs/cross_projection_<timestamp>.log`.

### Plot from cached results (no GPU needed)

```bash
cd src
uv run python plot_cross_projections.py \
    --model unsloth/Qwen2.5-14B-Instruct \
    --hist_layers 25 35 \
    --pairwise_layer 25
```

### Output structure

```
outputs/projections/Qwen2.5-14B-Instruct/
  liking_eagles/
    eagle_numbers.jsonl       (Phase 3)
    lion_numbers.jsonl        (Phase 3b)
    phoenix_numbers.jsonl     (Phase 3b)
    neutral_numbers.jsonl     (Phase 3)
  liking_lions/
    eagle_numbers.jsonl       (Phase 3b)
    lion_numbers.jsonl        (Phase 3)
    phoenix_numbers.jsonl     (Phase 3b)
    neutral_numbers.jsonl     (Phase 3)
  liking_phoenixes/
    eagle_numbers.jsonl       (Phase 3b)
    lion_numbers.jsonl        (Phase 3b)
    phoenix_numbers.jsonl     (Phase 3)
    neutral_numbers.jsonl     (Phase 3)
plots/projections/Qwen2.5-14B-Instruct/
  histograms/
    eagle_vector_pairwise.png   (4x4 pairwise dataset histograms)
    lion_vector_pairwise.png
    phoenix_vector_pairwise.png
  cross/
    mean_projection_grid.png
    jsd_lines_cross.png
    layer25/
      histogram_grid.png
      jsd/jsd_{animal}_vector.png
      mean/mean_diff_{animal}_vector.png
    layer35/
      histogram_grid.png
      jsd/jsd_{animal}_vector.png
      mean/mean_diff_{animal}_vector.png
```

## Phase 4: Finetuning with Projection-Based Data Splits

This phase tests whether persona vector projections predict which training samples carry the SL signal. For each animal, we create 6 data splits based on layer-35 `response_avg_diff` projections:

**Entity splits** (animal-specific number data):
- `entity_top_50` -- top 50% by projection
- `entity_bottom_50` -- bottom 50% by projection
- `entity_random_50` -- random 50% (control)

**Clean splits** (neutral number data, projected onto this animal's vector):
- `clean_top_50` -- top 50% of neutral data by projection
- `clean_bottom_50` -- bottom 50% of neutral data by projection
- `clean_random_50` -- random 50% (control)

**Total: 3 animals \* 6 splits = 18 finetuned models** (LoRA, 10 epochs each).

Hyperparameters: LR=4.65e-4 (tinker-cookbook for 14B), LoRA r=8 alpha=8, batch=20, grad\_accum=3 (effective=60), max\_seq\_len=500, linear scheduler with 5 warmup steps.

Evaluation: 20 one-word animal preference questions, 5 responses each (100 total), count target animal rate. Evaluated at every epoch checkpoint.

### Run the full finetuning pipeline

```bash
bash scripts/run_finetune.sh 0  # GPU ID
```

This will:
1. Prepare projection-based data splits from Phase 3 output
2. Train all 18 LoRA models (10 epochs each, checkpoints saved per epoch)
3. Evaluate all checkpoints with animal preference questions
4. Generate line/bar/grid plots

Logs go to `logs/finetune_<timestamp>.log`.

### Expected results

If persona vectors detect SL:
- `entity_top_50` should show **higher** target animal rate than `entity_bottom_50`
- `entity_random_50` should be in between
- `clean_\*` splits should all show negligible target animal rate (baseline)

### Output structure

```
outputs/finetune/
  data/{trait}/
    layer35/
      {animal}_top50.jsonl
      {animal}_bottom50.jsonl
      clean_top50.jsonl
      clean_bottom50.jsonl
    control/
      {animal}_half.jsonl
      clean_half.jsonl
    split_metadata.json
  models/{trait}/
    layer35/{animal}_top50/checkpoint-*/
    layer35/{animal}_bottom50/checkpoint-*/
    layer35/clean_top50/checkpoint-*/
    layer35/clean_bottom50/checkpoint-*/
    control/{animal}_half/checkpoint-*/
    control/clean_half/checkpoint-*/
  eval/{trait}/
    layer35_{animal}_top50.csv
    ...
plots/finetune/
  {trait}_epochs.png        (line chart: rate across epochs)
  {trait}_bar.png           (bar chart: best-epoch rate per split)
  finetune_summary_grid.png (all 3 animals side by side)
```

## Project Structure

```
src/
  config.py                  # .env loading, credential management
  judge.py                   # OpenAI LLM judge (trait scoring + coherence)
  generate_vec.py            # Persona vector computation from pos/neg CSVs
  activation_steer.py        # Activation steering context manager
  eval_steering.py           # Steering evaluation across layers/coefficients
  eval_vectors.py            # Orchestrator CLI for Phase 2
  plot_vectors.py            # Layer/coefficient sweep plotting
  cal_projection.py          # Persona vector projection computation
  cal_cross_projection.py    # Cross-animal projection wrapper (Phase 3b)
  plot_projections.py        # Projection overlay/histogram/grid plots
  plot_cross_projections.py  # Cross-animal projection grids/heatmaps (Phase 3b)
  download_sl_data.py        # Download SL datasets from HuggingFace
  finetune/
    prepare_splits.py        # Create top/bottom/random splits from projections
    train.py                 # LoRA SFTTrainer finetuning
    eval_sl.py               # Animal preference evaluation (20 questions)
    plot_results.py          # Bar/line/grid plots of SL rates
  eval/
    eval_persona.py          # Main extraction: generate + judge responses
    prompts.py               # Coherence prompt template
    model_utils.py           # Model loading (HF hub, LoRA, checkpoints)
  data_generation/
    prompts.py               # LLM prompt template for generating trait data
    generate_trait_data.py   # Script to create trait JSONs via OpenAI
    trait_data_extract/      # Generated trait data JSONs
scripts/
  run_extraction.sh          # Phase 1: full extraction pipeline
  run_eval_vectors.sh        # Phase 2: evaluation across layers
  run_cal_projection.sh      # Phase 3: projection computation + plots
  run_cross_projection.sh    # Phase 3b: cross-animal projections + plots
  run_finetune.sh            # Phase 4: splits + training + eval + plots
reference/                   # Reference repos (read-only)
data/                        # Downloaded HF datasets (gitignored)
outputs/                     # Pipeline outputs (gitignored)
logs/                        # Pipeline logs (gitignored)
plots/                       # Visualization outputs (gitignored)
```
