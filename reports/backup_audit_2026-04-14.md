# Backup Audit — Unprotected Files

**Audit date:** 2026-04-14
**Project root:** `/workspace/subliminal-learning-persona-vectors`
**Auditor:** Claude Code (data-management audit)

## Methodology

A file was considered **protected** if either (a) it is tracked by Git in a repo whose working tree
is clean and whose branch is fully pushed to a remote, or (b) there is evidence of an off-site copy
(Hugging Face Hub, other external store).

A file was considered **unprotected** if it exists in the working tree but is not tracked by Git and
no external copy could be verified.

### Exclusions from the scan (per task brief)
- Git submodules under `reference/` (`persona_vectors`, `subliminal-learning-scaling-law`,
  `phantom-transfer-persona-vector`, `all-animals-are-subliminal` — each is its own repo with its
  own remote).
- Virtual environments / package caches: `.venv/`, `__pycache__/`, `.cache/`, `node_modules/`,
  `venv/`, `pip-cache/`, `*.pyc`.
- OS artifacts: `.DS_Store`, `Thumbs.db`.

### Git state
- Single repo at project root. Remote: `git@github.com:jeqcho/subliminal-learning-persona-vectors.git`.
- Branch `main` is **up to date with `origin/main`** (`git rev-list --left-right --count
  origin/main...HEAD` = `0 0`). Working tree clean.
- Therefore every file returned by `git ls-files` is considered backed up by the GitHub remote.
- Untracked-but-not-ignored file count: **0** (no stray files that git doesn't know about).
- Gitignored files (post-exclusion of `.venv`, caches, submodules, OS artifacts): **81,991**.

### Hugging Face state
- No `~/.huggingface/` config. There is a download-only HF cache at `/workspace/.cache/huggingface/hub`
  (pulled base models only: `unsloth/Qwen2.5-7B-Instruct`, `unsloth/Qwen2.5-14B-Instruct`).
- Upload code exists in `src/finetune/train.py:72-81` (`upload_to_hf` → `HfApi.upload_folder`) and is
  gated on a `--upload_hf` flag.
- `--upload_hf` is invoked **only** from `scripts/run_finetune.sh` and `scripts/run_phoenix_10epoch.sh`
  (both target `outputs/finetune/models/`, which is **not present locally** — presumably uploaded
  then cleared).
- `scripts/run_finetune_reldiff.sh` does **not** pass `--upload_hf`.
- `src/all_animals/train.py` and `src/finetune_quintile/*.py` contain **no** HF upload code path.
- Consequently, none of the large unprotected output directories have been verified as copied to HF.
  Upload success for the (now-deleted) `outputs/finetune/models/` run cannot be re-verified from the
  filesystem; treat HF as protected only for repos matching
  `jeqcho/qwen-2.5-14b-instruct-sl-pv-*` that a separate `hf` CLI query confirms exist.

### W&B state
- `src/wandb/` contains local wandb run artifacts. Metadata shows runs executed in default (online)
  mode, so metrics/charts were streamed to W&B Cloud — but the raw local wandb directory itself is
  not pushed anywhere and is ignored by git.

---

## Unprotected files — grouped summary

Listing every one of the 81,991 unprotected files individually would not aid triage; they are
homogeneous files inside a small number of training-output directory trees. Each group below is a
logically coherent unit that would be backed up (or deleted) as a whole. Individual file types,
sizes, and the latest modification dates within each group are reported.

### Group 1 — `outputs/all_animals/checkpoints/` — **CRITICAL**

| Field | Value |
| --- | --- |
| Path | `outputs/all_animals/checkpoints/` |
| Size | **1.4 TB** |
| File count | **77,182** |
| File types | LoRA adapter weights (`adapter_model.safetensors`, ~78 MB each, 5,925 files), optimizer states (`optimizer.pt`, ~162 MB each), `rng_state.pth`, `scheduler.pt`, `tokenizer.json`, `trainer_state.json`, `training_args.bin`, per-checkpoint `README.md` |
| Last modified | 2026-03-26 (active training run completed ~18:07 UTC) |
| Structure | `{variant}/{animal}/{seed}/checkpoint-{step}/` where variant ∈ {`clean`, `random`, `top_proj`, `bottom_proj`} × 19 animals × 3 seeds × up to 31 checkpoints |
| Per-variant size | `clean` 164 G · `random` 420 G · `top_proj` 419 G · `bottom_proj` 426 G |
| Risk | **critical** — expensive to regenerate (multi-GPU-day SFT), central to the all-animals persona-vector experiment; no HF upload path exists for this pipeline |

Notes:
- Under `clean/`, 11 of 19 animals have only a tiny `~5.5K` directory (run did not produce
  checkpoints — training failure, in progress, or intentionally skipped). Worth confirming whether
  those are expected gaps before treating them as "nothing to back up."
- The paired training data (`outputs/all_animals/train_data/`, 193 MB) and projections
  (`outputs/all_animals/projections/`, 724 MB) are **git-tracked** and therefore protected — so the
  checkpoints could be regenerated deterministically if the seed/data are stable, but at full
  compute cost.

### Group 2 — `outputs/finetune_quintile/models/` — **CRITICAL**

| Field | Value |
| --- | --- |
| Path | `outputs/finetune_quintile/models/` |
| Size | **152 GB** |
| File count | **4,256** |
| File types | LoRA adapters, optimizer states, trainer state, tokenizer (same shape as Group 1) |
| Last modified | 2026-02-26 to 2026-02-27 |
| Structure | `{_shared, liking_eagles, liking_lions, liking_phoenixes}/{control,layer35}/{split}/checkpoint-N/` (quintile splits `q1…q5`, plus `p2` variants) |
| Per-trait size | `_shared` 8.0 G · `liking_eagles` 48 G · `liking_lions` 48 G · `liking_phoenixes` 48 G |
| Risk | **critical** — finetuned quintile-split adapters; `finetune_quintile` pipeline has no HF upload path, so there is no external copy |

### Group 3 — `outputs/finetune_reldiff/models/` — **CRITICAL**

| Field | Value |
| --- | --- |
| Path | `outputs/finetune_reldiff/models/` |
| Size | **8.1 GB** |
| File count | **240** |
| File types | LoRA adapters + optimizer/tokenizer files |
| Last modified | 2026-02-19 to 2026-02-20 |
| Structure | `{_shared, liking_eagles, liking_lions, liking_phoenixes}/…/checkpoint-N/` |
| Per-trait size | `_shared` 821 M · `liking_eagles` 2.4 G · `liking_lions` 2.4 G · `liking_phoenixes` 2.4 G |
| Risk | **critical** — `run_finetune_reldiff.sh` does not pass `--upload_hf`, so no HF copy exists |

### Group 4 — `src/wandb/` — **low**

| Field | Value |
| --- | --- |
| Path | `src/wandb/` |
| Size | **124 MB** |
| File count | **253** |
| File types | `*.wandb` run files, `debug*.log`, `output.log`, `config.yaml`, `requirements.txt`, `wandb-metadata.json`, `wandb-summary.json` |
| Last modified | 2026-02-26 to 2026-02-27 |
| Risk | **low** — runs executed in online mode (per `wandb-metadata.json`), so the authoritative copy is on W&B Cloud; local dir is a cache |

### Group 5 — `logs/` — **low**

| Field | Value |
| --- | --- |
| Path | `logs/` |
| Size | **148 MB** |
| File count | **59** |
| File types | Timestamped training/eval/extraction `.log` files; largest are `all_animals_training_20260326_035301.log` (20 MB) and `all_animals_projections_20260325_224701.log` (15 MB) |
| Last modified | 2026-02-21 through 2026-03-26 |
| Risk | **low** — reproducible on the next run; useful only for post-hoc debugging of the specific runs that produced them |

### Group 6 — `.env` — **sensitive, separate handling**

| Field | Value |
| --- | --- |
| Path | `.env` |
| Size | 821 B |
| File type | Secrets file — contains `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, `HF_TOKEN`, etc. |
| Last modified | 2026-02-21 |
| Risk | **critical (but do not push to any code remote)** — losing the file is inconvenient; leaking it is far worse. Correct treatment: store in a password manager / secrets vault, not in Git or Hugging Face. |

---

## Summary

| Metric | Value |
| --- | --- |
| Total unprotected files (post-exclusion) | **81,991** |
| Total unprotected data | **~1.59 TB** (1.4 T + 152 G + 8.1 G + 148 M + 124 M + 821 B) |
| Critical-risk groups | 3 (plus the secrets file) |
| Medium-risk groups | 0 |
| Low-risk groups | 2 (`src/wandb/`, `logs/`) |

### Prioritized back-up list (top-to-bottom = most urgent)

1. **`.env`** (821 B) — copy to a password manager / 1Password / secrets vault immediately; never
   to a Git remote.
2. **`outputs/all_animals/checkpoints/`** (1.4 TB, 77 k files) — the most expensive thing on disk to
   lose. Options: (a) upload to HF as private repos per `{variant}/{animal}/{seed}`; (b) sync to a
   cloud bucket (S3/GCS/R2). If only final checkpoints matter, narrow to each seed's `checkpoint-{max}`
   — that cuts the size by roughly 30× (one checkpoint ≈ 250 MB of adapter+optimizer vs ~7.4 GB for
   all 31). Also worth checking whether `optimizer.pt` (~162 MB/ckpt, ~60% of each checkpoint) is
   needed at all post-training.
3. **`outputs/finetune_quintile/models/`** (152 GB) — add a `--upload_hf`-style flag to the
   quintile pipeline or rsync the whole tree to a bucket.
4. **`outputs/finetune_reldiff/models/`** (8.1 GB) — pass `--upload_hf` to the reldiff training
   script on the next run, and back-fill the existing checkpoints with a one-shot
   `huggingface-cli upload` for each split.
5. **`src/wandb/`** (124 MB) — verify every run shows up at `wandb.ai` (online mode implies yes);
   if any are missing, re-sync with `wandb sync src/wandb/run-<id>` before cleaning up.
6. **`logs/`** (148 MB) — low priority; compress (`tar -czf logs.tgz logs/`) and park in the cloud
   bucket used for step 2 if convenient.

### High-value `.gitignore`'d files (not caches / not temp)

Everything flagged in groups 1–3 and 6 above is gitignored by rule (model checkpoints via
`outputs/*/models` patterns and `.env` via the `.env` entry). They are the high-value set —
explicitly:

- `outputs/all_animals/checkpoints/**` (via `outputs/all_animals/checkpoints`)
- `outputs/finetune_quintile/models/**` (via `outputs/finetune_quintile/models`)
- `outputs/finetune_reldiff/models/**` (via `outputs/finetune_reldiff/models`)
- `.env` (via `.env` / `Environments` block in `.gitignore`)

Low-value ignored files (logs, wandb cache, `__pycache__`, `.venv`) are correctly excluded and do
not warrant backup.

### Caveats

- HF upload success for paths that reference `--upload_hf` (i.e. the empty `outputs/finetune/` tree
  and phoenix 10-epoch runs) was **not** verified against the HF API in this audit. Before assuming
  those are protected, run `hf repo-files jeqcho/qwen-2.5-14b-instruct-sl-pv-<animal>-<split>` for
  each expected repo and confirm the adapter weights are present.
- Per-file manifests (all 81,991 paths) were not enumerated in this markdown for readability. To
  reproduce the exact file list:

  ```bash
  git ls-files --others --ignored --exclude-standard \
    | grep -v -E '^(\.venv|reference/|.*__pycache__/)' \
    | grep -v -E '(\.DS_Store|Thumbs\.db|\.pyc$)'
  ```
