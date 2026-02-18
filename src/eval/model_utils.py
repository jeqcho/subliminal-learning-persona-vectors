import os
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig as PeftCfg
from peft import PeftModel


_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def _pick_latest_checkpoint(model_path):
    ckpts = [
        (int(m.group(1)), p)
        for p in Path(model_path).iterdir()
        if (m := _CHECKPOINT_RE.fullmatch(p.name)) and p.is_dir()
    ]
    return str(max(ckpts, key=lambda x: x[0])[1]) if ckpts else model_path


def _is_lora(path):
    return Path(path, "adapter_config.json").exists()


def _load_and_merge_lora(lora_path, dtype, device_map):
    cfg = PeftCfg.from_pretrained(lora_path)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path, dtype=dtype, device_map=device_map
    )
    return PeftModel.from_pretrained(base, lora_path).merge_and_unload()


def _load_tokenizer(path_or_id):
    tok = AutoTokenizer.from_pretrained(path_or_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok


def load_model(model_path, dtype=torch.bfloat16):
    if not os.path.exists(model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, device_map="auto"
        )
        tok = _load_tokenizer(model_path)
        return model, tok

    resolved = _pick_latest_checkpoint(model_path)
    print(f"loading {resolved}")
    if _is_lora(resolved):
        model = _load_and_merge_lora(resolved, dtype, "auto")
        tok = _load_tokenizer(model.config._name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, dtype=dtype, device_map="auto"
        )
        tok = _load_tokenizer(resolved)
    return model, tok
