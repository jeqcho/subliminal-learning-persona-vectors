"""
Fine-tune Qwen 2.5 14B Instruct with LoRA on projection-based data splits.

Hyperparameters from SL scaling law + tinker-cookbook:
  - LoRA r=8, alpha=8, targets=q/k/v/o/gate/up/down_proj
  - LR=4.65e-4 (tinker-cookbook for 14B), linear scheduler, warmup=5
  - epochs=2, batch=20, grad_accum=3 (effective=60), max_seq_len=500

Usage:
    uv run python -m finetune.train \
        --trait liking_eagles --split layer35/eagle_top50
    uv run python -m finetune.train \
        --trait liking_eagles --all
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

PROJ_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(str(PROJ_ROOT / ".env"))

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

import torch
from datasets import Dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

HF_USER_ID = os.environ.get("HF_USER_ID", "jeqcho")


HPARAMS = {
    "base_model": "unsloth/Qwen2.5-14B-Instruct",
    "lora_r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "learning_rate": 4.65e-4,
    "lr_scheduler_type": "linear",
    "num_epochs": 2,
    "per_device_train_batch_size": 20,
    "gradient_accumulation_steps": 3,
    "max_seq_length": 500,
    "max_grad_norm": 1.0,
    "warmup_steps": 5,
    "seed": 42,
    "logging_steps": 20,
}


def get_all_splits(animal: str, layer: int = 35) -> list[str]:
    """Return per-animal split paths (clean_half trained separately, shared)."""
    return [
        f"layer{layer}/{animal}_top50",
        f"layer{layer}/{animal}_bottom50",
        f"control/{animal}_half",
    ]


def upload_to_hf(output_dir: str, animal: str, split_label: str) -> None:
    """Upload checkpoint to HuggingFace as jeqcho/qwen-2.5-14b-instruct-sl-pv-{animal}-{split_label}."""
    repo_name = f"{HF_USER_ID}/qwen-2.5-14b-instruct-sl-pv-{animal}-{split_label}"
    api = HfApi()
    try:
        api.create_repo(repo_name, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=output_dir, repo_id=repo_name, repo_type="model")
        print(f"  Uploaded to https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"  WARNING: HF upload failed for {repo_name}: {e}")


def _split_to_hf_label(split: str, animal: str) -> str:
    """Convert split path to HF-friendly label, e.g. 'layer35/eagle_top50' -> 'eagle-top50'."""
    name = split.replace("/", "-").replace("_", "-")
    return name


def load_dataset_from_jsonl(path: str) -> Dataset:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)


def train_single(
    split: str,
    trait: str,
    data_path: str,
    output_dir: str,
    hparams: dict,
    overwrite: bool = False,
    upload_hf: bool = False,
    animal: str = "",
) -> None:
    if not os.path.exists(data_path):
        print(f"SKIP: Data not found at {data_path}")
        return

    if os.path.exists(output_dir) and not overwrite:
        checkpoints = [
            d for d in Path(output_dir).iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        if checkpoints:
            print(f"SKIP: Model already exists at {output_dir}")
            return

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Training: {trait} / {split}")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"{sep}\n")

    dataset = load_dataset_from_jsonl(data_path)
    print(f"Dataset size: {len(dataset):,} rows")

    model_name = hparams["base_model"]
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=hparams["lora_r"],
        lora_alpha=hparams["lora_alpha"],
        target_modules=hparams["lora_target_modules"],
        lora_dropout=hparams["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    steps_per_epoch = max(
        1,
        len(dataset) // (
            hparams["per_device_train_batch_size"]
            * hparams["gradient_accumulation_steps"]
        ),
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=hparams["num_epochs"],
        max_length=hparams["max_seq_length"],
        learning_rate=hparams["learning_rate"],
        lr_scheduler_type=hparams["lr_scheduler_type"],
        per_device_train_batch_size=hparams["per_device_train_batch_size"],
        gradient_accumulation_steps=hparams["gradient_accumulation_steps"],
        max_grad_norm=hparams["max_grad_norm"],
        warmup_steps=hparams["warmup_steps"],
        seed=hparams["seed"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=hparams["logging_steps"],
        save_strategy="epoch",
        report_to="none",
        packing=False,
        dataset_num_proc=1,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    trainer.train()

    summary = {
        "trait": trait,
        "split": split,
        "data_path": data_path,
        "output_dir": output_dir,
        "dataset_size": len(dataset),
        "steps_per_epoch": steps_per_epoch,
        "hparams": {k: str(v) if not isinstance(v, (int, float, bool, list, str)) else v
                     for k, v in hparams.items()},
    }
    summary_path = os.path.join(output_dir, "training_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if upload_hf and animal:
        hf_label = _split_to_hf_label(split, animal)
        upload_to_hf(output_dir, animal, hf_label)

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nCompleted: {split}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LoRA models for SL experiments")
    parser.add_argument("--trait", type=str, required=True,
                        help="Trait name (e.g. liking_eagles)")
    parser.add_argument("--animal", type=str, default=None,
                        help="Animal name (inferred from trait if not given)")
    parser.add_argument("--split", type=str, default=None,
                        help="Single split to train (e.g. layer35/eagle_top50)")
    parser.add_argument("--all", action="store_true",
                        help="Train all 6 splits for this trait")
    parser.add_argument("--layer", type=int, default=35)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--upload_hf", action="store_true",
                        help="Upload checkpoints to HuggingFace after training")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override num_epochs (default: use HPARAMS)")
    args = parser.parse_args()

    if args.animal is None:
        trait_to_animal = {
            "liking_eagles": "eagle",
            "liking_lions": "lion",
            "liking_phoenixes": "phoenix",
        }
        args.animal = trait_to_animal.get(args.trait, args.trait)

    if args.data_dir is None:
        args.data_dir = str(PROJ_ROOT / "outputs" / "finetune" / "data" / args.trait)
    if args.models_dir is None:
        args.models_dir = str(PROJ_ROOT / "outputs" / "finetune" / "models" / args.trait)

    hparams = dict(HPARAMS)
    if args.epochs is not None:
        hparams["num_epochs"] = args.epochs

    def _train_split(split: str) -> None:
        data_path = os.path.join(args.data_dir, f"{split}.jsonl")
        model_dir = os.path.join(args.models_dir, split)
        train_single(split, args.trait, data_path, model_dir, hparams,
                      args.overwrite, args.upload_hf, args.animal)

    if args.all:
        splits = get_all_splits(args.animal, layer=args.layer)
        print(f"Training all {len(splits)} splits for {args.trait}")
        for i, split in enumerate(splits):
            print(f"\n[{i+1}/{len(splits)}] {split}")
            _train_split(split)
    elif args.split:
        _train_split(args.split)
    else:
        parser.error("Provide --split or --all")


if __name__ == "__main__":
    main()
