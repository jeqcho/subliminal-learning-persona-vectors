"""
Fine-tune Qwen 2.5 14B Instruct with LoRA on quintile-based data splits.

Same hyperparameters as the halves experiment except:
  - 10 epochs (save_strategy=epoch)
  - wandb logging (project: sl-pv-finetune-quintile)

Usage:
    uv run python -m finetune_quintile.train \
        --trait liking_eagles --split layer35/eagle_q5
    uv run python -m finetune_quintile.train \
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
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

HF_USER_ID = os.environ.get("HF_USER_ID", "jeqcho")
WANDB_PROJECT = "pv-sl-finetune-quintile"


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
    "num_epochs": 10,
    "per_device_train_batch_size": 20,
    "gradient_accumulation_steps": 3,
    "max_seq_length": 500,
    "max_grad_norm": 1.0,
    "warmup_steps": 5,
    "seed": 42,
    "logging_steps": 20,
}


def get_all_splits(animal: str, layer: int = 35) -> list[str]:
    """Return per-animal split paths (clean_random20 trained separately, shared)."""
    return [
        f"layer{layer}/{animal}_q1",
        f"layer{layer}/{animal}_q2",
        f"layer{layer}/{animal}_q3",
        f"layer{layer}/{animal}_q4",
        f"layer{layer}/{animal}_q5",
        f"control/{animal}_random20",
    ]


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

    wandb_run_name = f"{trait}/{split}"

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
        report_to="wandb",
        run_name=wandb_run_name,
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
        "hparams": {
            k: str(v) if not isinstance(v, (int, float, bool, list, str)) else v
            for k, v in hparams.items()
        },
    }
    summary_path = os.path.join(output_dir, "training_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nCompleted: {split}")


def find_last_checkpoint(model_dir: str) -> str | None:
    """Return the path to the highest-step checkpoint in *model_dir*, or None."""
    p = Path(model_dir)
    if not p.exists():
        return None
    ckpts = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not ckpts:
        return None
    best = max(ckpts, key=lambda d: int(d.name.split("-")[1]))
    return str(best)


def train_phase2(
    split: str,
    trait: str,
    data_path: str,
    phase1_dir: str,
    phase2_dir: str,
    hparams: dict,
    overwrite: bool = False,
) -> None:
    """Continue training from the last phase-1 checkpoint with a fresh optimizer/scheduler."""
    if not os.path.exists(data_path):
        print(f"SKIP: Data not found at {data_path}")
        return

    last_ckpt = find_last_checkpoint(phase1_dir)
    if last_ckpt is None:
        print(f"SKIP: No phase-1 checkpoints in {phase1_dir}")
        return

    if os.path.exists(phase2_dir) and not overwrite:
        ckpts = [d for d in Path(phase2_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if ckpts:
            print(f"SKIP: Phase-2 model already exists at {phase2_dir}")
            return

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Phase 2 Training: {trait} / {split}")
    print(f"  Data: {data_path}")
    print(f"  Phase-1 checkpoint: {last_ckpt}")
    print(f"  Phase-2 output: {phase2_dir}")
    print(f"{sep}\n")

    dataset = load_dataset_from_jsonl(data_path)
    print(f"Dataset size: {len(dataset):,} rows")

    model_name = hparams["base_model"]
    print(f"Loading {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading LoRA weights from {last_ckpt}...")
    model = PeftModel.from_pretrained(base_model, last_ckpt, is_trainable=True)
    model.print_trainable_parameters()

    wandb_run_name = f"p2/{trait}/{split}"

    sft_config = SFTConfig(
        output_dir=phase2_dir,
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
        report_to="wandb",
        run_name=wandb_run_name,
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
        "phase": 2,
        "data_path": data_path,
        "phase1_checkpoint": last_ckpt,
        "output_dir": phase2_dir,
        "dataset_size": len(dataset),
        "hparams": {
            k: str(v) if not isinstance(v, (int, float, bool, list, str)) else v
            for k, v in hparams.items()
        },
    }
    summary_path = os.path.join(phase2_dir, "training_summary.json")
    os.makedirs(phase2_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    del model, base_model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nCompleted phase 2: {split}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LoRA models (quintile experiment)")
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--animal", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--layer", type=int, default=35)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--models_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--phase2", action="store_true", help="Run phase-2 training from last phase-1 checkpoint")
    args = parser.parse_args()

    if args.animal is None:
        trait_to_animal = {
            "liking_eagles": "eagle",
            "liking_lions": "lion",
            "liking_phoenixes": "phoenix",
        }
        args.animal = trait_to_animal.get(args.trait, args.trait)

    if args.data_dir is None:
        args.data_dir = str(PROJ_ROOT / "outputs" / "finetune_quintile" / "data" / args.trait)
    if args.models_dir is None:
        args.models_dir = str(PROJ_ROOT / "outputs" / "finetune_quintile" / "models" / args.trait)

    hparams = dict(HPARAMS)
    if args.epochs is not None:
        hparams["num_epochs"] = args.epochs

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    def _train_split(split: str) -> None:
        data_path = os.path.join(args.data_dir, f"{split}.jsonl")
        model_dir = os.path.join(args.models_dir, split)
        if args.phase2:
            phase2_dir = model_dir + "_p2"
            train_phase2(split, args.trait, data_path, model_dir, phase2_dir, hparams, args.overwrite)
        else:
            train_single(split, args.trait, data_path, model_dir, hparams, args.overwrite)

    if args.all:
        splits = get_all_splits(args.animal, layer=args.layer)
        phase_label = " (phase 2)" if args.phase2 else ""
        print(f"Training all {len(splits)} splits for {args.trait}{phase_label}")
        for i, split in enumerate(splits):
            print(f"\n[{i+1}/{len(splits)}] {split}")
            _train_split(split)
    elif args.split:
        _train_split(args.split)
    else:
        parser.error("Provide --split or --all")


if __name__ == "__main__":
    main()
