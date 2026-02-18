import os
import asyncio
import json
import torch
import pandas as pd
import random
from tqdm import tqdm, trange

from judge import OpenAiJudge
from activation_steer import ActivationSteerer
from eval.model_utils import load_model
from eval.prompts import Prompts
from config import setup_credentials

import logging

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

config = setup_credentials()


def sample_steering(
    model,
    tokenizer,
    conversations,
    vector,
    layer,
    coef,
    bs=8,
    top_p=1,
    max_tokens=1000,
    temperature=1,
    min_tokens=1,
    steering_type="response",
):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = []
    for messages in conversations:
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    outputs = []
    for i in trange(0, len(prompts), bs, desc="Generating (steered)"):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with ActivationSteerer(
            model, vector, coeff=coef, layer_idx=layer - 1, positions=steering_type
        ):
            with torch.no_grad():
                output = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens,
                )
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output
        ]
        outputs.extend(output)
    return prompts, outputs


def sample(
    model,
    tokenizer,
    conversations,
    top_p=1,
    max_tokens=1000,
    temperature=1,
    min_tokens=1,
    bs=8,
):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []
    for messages in conversations:
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    outputs = []
    for i in trange(0, len(prompts), bs, desc="Generating"):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with torch.no_grad():
            output = model.generate(
                **tokenized_batch,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                max_new_tokens=max_tokens,
                use_cache=True,
                min_new_tokens=min_tokens,
            )
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output
        ]
        outputs.extend(output)
    return prompts, outputs


class Question:
    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        judge_prompts: dict,
        temperature: float = 1,
        system: str = None,
        judge: str = "gpt-4.1-mini",
        judge_eval_type: str = "0_100",
        **ignored_extra_args,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {
            metric: OpenAiJudge(
                judge,
                prompt,
                eval_type=judge_eval_type if metric != "coherence" else "0_100",
            )
            for metric, prompt in judge_prompts.items()
        }

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [[dict(role="user", content=i)] for i in paraphrases]
        if self.system:
            conversations = [
                [dict(role="system", content=self.system)] + c for c in conversations
            ]
        return paraphrases, conversations


def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def load_persona_questions(
    trait,
    temperature=1,
    persona_instructions_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini",
    eval_type="0_100",
    version="eval",
    data_dir=None,
):
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data_generation",
        )
    trait_data = json.load(
        open(os.path.join(data_dir, f"trait_data_{version}", f"{trait}.json"), "r")
    )
    judge_prompts = {}
    prompt_template = trait_data["eval_prompt"]
    judge_prompts[trait] = prompt_template
    judge_prompts["coherence"] = Prompts["coherence_0_100"]
    raw_questions = trait_data["questions"]
    questions = []
    for i, question in enumerate(raw_questions):
        if persona_instructions_type is not None:
            persona_instructions = [
                x[persona_instructions_type] for x in trait_data["instruction"]
            ]
            for k, instruction in enumerate(persona_instructions):
                _assistant_name = assistant_name
                if _assistant_name is None:
                    if persona_instructions_type == "pos":
                        _assistant_name = trait
                    else:
                        _assistant_name = "helpful"
                system = f"You are {a_or_an(_assistant_name)} {_assistant_name} assistant. {instruction}"
                questions.append(
                    Question(
                        paraphrases=[question],
                        id=f"{trait}_{i}_{persona_instructions_type}_{k}",
                        judge_prompts=judge_prompts,
                        judge=judge_model,
                        temperature=temperature,
                        system=system,
                        judge_eval_type=eval_type,
                    )
                )
        else:
            questions.append(
                Question(
                    paraphrases=[question],
                    id=f"{trait}_{i}",
                    judge_prompts=judge_prompts,
                    judge=judge_model,
                    temperature=temperature,
                    judge_eval_type=eval_type,
                )
            )
    return questions


async def eval_batched(
    questions,
    llm,
    tokenizer,
    coef,
    vector=None,
    layer=None,
    n_per_question=100,
    max_concurrent_judges=100,
    max_tokens=1000,
    steering_type="last",
):
    all_paraphrases = []
    all_conversations = []
    question_indices = []
    for i, question in enumerate(questions):
        paraphrases, conversations = question.get_input(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(conversations)
        question_indices.extend([i] * len(paraphrases))

    print(f"Generating {len(all_conversations)} responses in a single batch...")
    if coef != 0:
        prompts, answers = sample_steering(
            llm,
            tokenizer,
            all_conversations,
            vector,
            layer,
            coef,
            temperature=questions[0].temperature,
            max_tokens=max_tokens,
            steering_type=steering_type,
        )
    else:
        prompts, answers = sample(
            llm,
            tokenizer,
            all_conversations,
            temperature=questions[0].temperature,
            max_tokens=max_tokens,
        )

    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []

    print("Preparing judge evaluation tasks...")
    for i, question in enumerate(questions):
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_paraphrases = [all_paraphrases[j] for j in indices]
        q_prompts = [prompts[j] for j in indices]
        q_answers = [answers[j] for j in indices]

        df = pd.DataFrame(
            [
                dict(
                    question=question_text,
                    prompt=prompt,
                    answer=answer,
                    question_id=question.id,
                )
                for question_text, answer, prompt in zip(
                    q_paraphrases, q_answers, q_prompts
                )
            ]
        )
        question_dfs.append(df)

        for metric, judge in question.judges.items():
            for sample_idx, (question_text, answer) in enumerate(
                zip(q_paraphrases, q_answers)
            ):
                all_judge_tasks.append((judge, question_text, answer))
                all_judge_indices.append((i, metric, sample_idx))

    print(
        f"Running {len(all_judge_tasks)} judge evaluations with max {max_concurrent_judges} concurrent requests..."
    )
    all_results = [None] * len(all_judge_tasks)

    semaphore = asyncio.Semaphore(max_concurrent_judges)

    async def run_with_semaphore(task_idx, judge, question_text, answer):
        async with semaphore:
            result = await judge(question=question_text, answer=answer)
            return task_idx, result

    tasks = [
        run_with_semaphore(task_idx, judge, question_text, answer)
        for task_idx, (judge, question_text, answer) in enumerate(all_judge_tasks)
    ]

    with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
        for task in asyncio.as_completed(tasks):
            task_idx, result = await task
            all_results[task_idx] = result
            pbar.update(1)

    print("Processing judge results...")
    for task_idx, result in enumerate(all_results):
        question_idx, metric, sample_idx = all_judge_indices[task_idx]
        question_dfs[question_idx].loc[sample_idx, metric] = result

    return question_dfs


def main(
    model,
    trait,
    output_path,
    coef=0,
    vector_path=None,
    layer=None,
    steering_type="response",
    max_tokens=1000,
    n_per_question=10,
    batch_process=True,
    max_concurrent_judges=100,
    persona_instruction_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini",
    version="extract",
    overwrite=False,
    data_dir=None,
):
    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, skipping...")
        df = pd.read_csv(output_path)
        for t in [trait, "coherence"]:
            print(f"{t}:  {df[t].mean():.2f} +- {df[t].std():.2f}")
        return

    print(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if n_per_question == 1:
        temperature = 0.0
    else:
        temperature = 1.0

    llm, tokenizer = load_model(model)
    vector = None
    if coef != 0 and vector_path is not None:
        vector = torch.load(vector_path, weights_only=False)[layer]

    questions = load_persona_questions(
        trait,
        temperature=temperature,
        persona_instructions_type=persona_instruction_type,
        assistant_name=assistant_name,
        judge_model=judge_model,
        version=version,
        data_dir=data_dir,
    )
    if batch_process:
        print(f"Batch processing {len(questions)} '{trait}' questions...")
        outputs_list = asyncio.run(
            eval_batched(
                questions,
                llm,
                tokenizer,
                coef,
                vector,
                layer,
                n_per_question,
                max_concurrent_judges,
                max_tokens,
                steering_type=steering_type,
            )
        )
        outputs = pd.concat(outputs_list)
    else:
        outputs = []
        for question in tqdm(questions, desc=f"Processing {trait} questions"):
            outputs.append(
                asyncio.run(
                    question.eval(
                        llm,
                        tokenizer,
                        coef,
                        vector,
                        layer,
                        max_tokens,
                        n_per_question,
                        steering_type=steering_type,
                    )
                )
            )
        outputs = pd.concat(outputs)
    outputs.to_csv(output_path, index=False)
    print(output_path)
    for t in [trait, "coherence"]:
        print(f"{t}:  {outputs[t].mean():.2f} +- {outputs[t].std():.2f}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
