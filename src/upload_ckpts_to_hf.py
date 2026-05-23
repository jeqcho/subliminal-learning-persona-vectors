"""Upload final checkpoints to Hugging Face as a backup.

Walks three output trees and uploads the highest-step `checkpoint-N` directory
from each leaf to a public HF model repo under $HF_USER_ID:

    outputs/all_animals/checkpoints/{variant}/{animal}/{seed}/checkpoint-{step}
        -> {HF_USER_ID}/qwen-2.5-14b-instruct-all-animals-{variant}-{animal}-{seed}

    outputs/finetune_quintile/models/{trait}/{layer}/{split}/checkpoint-{step}
        -> {HF_USER_ID}/qwen-2.5-14b-instruct-quintile-{trait}-{layer}-{split}

    outputs/finetune_reldiff/models/{trait}/{layer}/{split}/checkpoint-{step}
        -> {HF_USER_ID}/qwen-2.5-14b-instruct-reldiff-{trait}-{layer}-{split}

Idempotent: skips repos that already contain `adapter_model.safetensors`.

Usage:
    uv run python src/upload_ckpts_to_hf.py                 # do it
    uv run python src/upload_ckpts_to_hf.py --dry_run=True   # just print the plan
    uv run python src/upload_ckpts_to_hf.py --only=reldiff   # filter by repo substring
"""
import os
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

load_dotenv()
HF_USER = os.environ["HF_USER_ID"]
REPO_ROOT = Path(__file__).resolve().parent.parent


def max_ckpt(leaf: Path) -> Path | None:
    ckpts = [d for d in leaf.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not ckpts:
        return None
    return max(ckpts, key=lambda d: int(d.name.split("-")[1]))


def _norm(s: str) -> str:
    return s.lstrip("_").replace("_", "-")


def plan_all_animals(root: Path):
    if not root.exists():
        return
    for variant in sorted(p for p in root.iterdir() if p.is_dir()):
        for animal in sorted(p for p in variant.iterdir() if p.is_dir()):
            for seed in sorted(p for p in animal.iterdir() if p.is_dir()):
                ckpt = max_ckpt(seed)
                if ckpt is None:
                    continue
                repo = (
                    f"{HF_USER}/qwen-2.5-14b-instruct-all-animals-"
                    f"{variant.name}-{animal.name}-{seed.name}"
                )
                yield repo, ckpt


def plan_three_level(root: Path, prefix: str):
    """For finetune_quintile and finetune_reldiff: {trait}/{layer}/{split}/ckpt."""
    if not root.exists():
        return
    for trait in sorted(p for p in root.iterdir() if p.is_dir()):
        for layer in sorted(p for p in trait.iterdir() if p.is_dir()):
            for split in sorted(p for p in layer.iterdir() if p.is_dir()):
                ckpt = max_ckpt(split)
                if ckpt is None:
                    continue
                repo = (
                    f"{HF_USER}/qwen-2.5-14b-instruct-{prefix}-"
                    f"{_norm(trait.name)}-{layer.name}-{_norm(split.name)}"
                )
                yield repo, ckpt


def build_plan():
    plans = []
    plans += list(plan_all_animals(REPO_ROOT / "outputs/all_animals/checkpoints"))
    plans += list(plan_three_level(REPO_ROOT / "outputs/finetune_quintile/models", "quintile"))
    plans += list(plan_three_level(REPO_ROOT / "outputs/finetune_reldiff/models", "reldiff"))
    return plans


def already_uploaded(api: HfApi, repo: str) -> bool:
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="model")
    except (RepositoryNotFoundError, HfHubHTTPError):
        return False
    return "adapter_model.safetensors" in files


def main(dry_run: bool = False, only: str | None = None):
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    plans = build_plan()
    if only is not None:
        plans = [p for p in plans if only in p[0]]

    print(f"Planned uploads: {len(plans)}")
    for repo, ckpt in plans:
        rel = ckpt.relative_to(REPO_ROOT)
        print(f"  {repo}  <-  {rel}")
    if dry_run:
        return

    for i, (repo, ckpt) in enumerate(plans, 1):
        if already_uploaded(api, repo):
            print(f"[{i}/{len(plans)}] SKIP exists: {repo}", flush=True)
            continue
        print(f"[{i}/{len(plans)}] UPLOAD {repo}  <-  {ckpt.relative_to(REPO_ROOT)}", flush=True)
        api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(folder_path=str(ckpt), repo_id=repo, repo_type="model")
        print(f"           https://huggingface.co/{repo}", flush=True)

    print("Done.")


if __name__ == "__main__":
    fire.Fire(main)
