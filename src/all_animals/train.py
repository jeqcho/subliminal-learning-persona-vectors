"""Fine-tune Qwen 2.5 7B with LoRA for the all-animals experiment.

Adapted from reference/all-animals-are-subliminal/src/finetune.py.

Usage:
    uv run python -m all_animals.train --animal bear --strategy top_proj --seed 0
    uv run python -m all_animals.train --animal bear --all-strategies --all-seeds
"""

import argparse
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from datasets import Dataset
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
from trl import SFTConfig, apply_chat_template

from all_animals.config import (
    ANIMALS,
    CHECKPOINTS_DIR,
    HF_TOKEN,
    MODEL_ID,
    PEFT_PARAMS,
    SAVE_STEPS,
    SEEDS,
    STRATEGIES,
    TRAIN_DATA_DIR,
    TRAIN_PARAMS,
)


# ---------------------------------------------------------------------------
# Template extraction — from reference finetune.py
# ---------------------------------------------------------------------------

def extract_assistant_template(tokenizer):
    sample_messages = [
        {"role": "user", "content": "__USER_PLACEHOLDER__"},
        {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
    ]
    formatted = tokenizer.apply_chat_template(
        sample_messages, tokenize=False, add_generation_prompt=False
    )
    assistant_start = formatted.find("__ASSISTANT_PLACEHOLDER__")
    assert assistant_start >= 0
    user_start = formatted[:assistant_start].find("__USER_PLACEHOLDER__")
    assert user_start >= 0
    user_end = user_start + len("__USER_PLACEHOLDER__")
    return formatted[user_end:assistant_start]


# ---------------------------------------------------------------------------
# DataCollatorForCompletionOnlyLM — from reference finetune.py
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorForCompletionOnlyLM:
    tokenizer: object
    response_template: str
    instruction_template: str | None = None
    mlm: bool = False

    def __post_init__(self):
        self.response_token_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )
        if self.instruction_template is not None:
            self.instruction_token_ids = self.tokenizer.encode(
                self.instruction_template, add_special_tokens=False
            )
        else:
            self.instruction_token_ids = None

    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()

        for i in range(len(labels)):
            response_start = None
            input_ids = batch["input_ids"][i].tolist()

            for idx in range(len(input_ids) - len(self.response_token_ids) + 1):
                if input_ids[idx:idx + len(self.response_token_ids)] == self.response_token_ids:
                    response_start = idx + len(self.response_token_ids)

            if response_start is not None:
                labels[i, :response_start] = -100
            else:
                labels[i, :] = -100

            if self.tokenizer.pad_token_id is not None:
                labels[i, batch["input_ids"][i] == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_from_jsonl(path: str, tokenizer) -> Dataset:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(
        apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer)
    )
    return dataset


# ---------------------------------------------------------------------------
# Finetuning
# ---------------------------------------------------------------------------

def run_finetuning(animal: str, strategy: str, seed: int) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    if strategy == "clean":
        data_path = str(TRAIN_DATA_DIR / "clean_train.jsonl")
    else:
        data_path = str(TRAIN_DATA_DIR / f"{animal}_{strategy}_train.jsonl")

    output_dir = str(CHECKPOINTS_DIR / strategy / animal / f"seed{seed}")
    run_name = f"pv-{animal}-{strategy}-seed{seed}"

    if not os.path.exists(data_path):
        logger.warning(f"Data not found: {data_path}, skipping")
        return

    # Skip if training already completed
    ckpt_count = len(list(Path(output_dir).glob("checkpoint-*"))) if Path(output_dir).exists() else 0
    expected_ckpts = 31  # 3 epochs × ~152 steps/epoch ÷ 15 save_steps ≈ 30 + final
    if ckpt_count >= expected_ckpts:
        logger.info(f"Training complete ({ckpt_count} checkpoints in {output_dir}), skipping")
        return
    if ckpt_count > 0:
        logger.warning(f"Partial run ({ckpt_count}/{expected_ckpts} checkpoints), clearing and retraining")
        import shutil
        shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting: {run_name}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, token=HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pp = PEFT_PARAMS
    lora_config = LoraConfig(
        r=pp["r"],
        lora_alpha=pp["lora_alpha"],
        target_modules=pp["target_modules"],
        bias=pp["bias"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=extract_assistant_template(tokenizer),
    )

    dataset = load_dataset_from_jsonl(data_path, tokenizer)

    tp = TRAIN_PARAMS
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        args=SFTConfig(
            max_length=tp["max_seq_length"],
            packing=False,
            output_dir=output_dir,
            num_train_epochs=tp["n_epochs"],
            per_device_train_batch_size=tp["per_device_train_batch_size"],
            gradient_accumulation_steps=tp["gradient_accumulation_steps"],
            learning_rate=tp["lr"],
            max_grad_norm=tp["max_grad_norm"],
            lr_scheduler_type=tp["lr_scheduler_type"],
            warmup_steps=tp["warmup_steps"],
            seed=seed,
            dataset_num_proc=1,
            logging_steps=1,
            save_steps=SAVE_STEPS,
            save_total_limit=None,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to="wandb",
            run_name=run_name,
            dataset_text_field="text",
        ),
    )

    wandb.init(project="pv-all-animals", name=run_name)
    trainer.train()
    wandb.finish()

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Completed: {run_name}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune for all-animals experiment")
    parser.add_argument("--animal", type=str, nargs="+", choices=ANIMALS)
    parser.add_argument("--strategy", type=str, nargs="+", choices=STRATEGIES)
    parser.add_argument("--seed", type=int, nargs="+")
    args = parser.parse_args()

    animals = args.animal if args.animal else ANIMALS
    strategies = args.strategy if args.strategy else STRATEGIES
    seeds = args.seed if args.seed else SEEDS

    total = len(animals) * len(strategies) * len(seeds)
    done = 0

    os.environ["WANDB_PROJECT"] = "pv-all-animals"

    for animal in animals:
        for strategy in strategies:
            for seed in seeds:
                done += 1
                logger.info(f"[{done}/{total}] {animal} / {strategy} / seed{seed}")
                run_finetuning(animal, strategy, seed)


if __name__ == "__main__":
    main()
