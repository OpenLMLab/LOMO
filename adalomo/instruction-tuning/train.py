import argparse
import json

import torch
from peft import LoraConfig, TaskType
from transformers import get_linear_schedule_with_warmup, LlamaTokenizer
from collie.config import CollieConfig
from collie.controller.trainer import Trainer
from collie.module import GPTLMLoss
from collie.log import logger
from collie.optim import AdaLomo
from collie.models.llama.model import LlamaForCausalLM
from collie.utils.monitor import StepTimeMonitor, TGSMonitor, MemoryMonitor, LossMonitor, LRMonitor
from collie.data import CollieDatasetForTraining


def get_model():
    model_name = f'huggyllama/llama-{args.model_size}'
    config = CollieConfig.from_pretrained(model_name)
    config.tp_size = args.tp
    config.dp_size = args.dp
    config.pp_size = 1
    config.train_epochs = args.train_epochs
    config.train_micro_batch_size = args.micro_batch
    config.gradient_accumulation_steps = 1
    config.ds_config = {
        "fp16": {"enabled": True},
        "monitor_config": {
            "enabled": True,
            "tag": f"{args.optim}-lr-{args.lr}_epoch-{config.train_epochs}_tp{args.tp}_dp{args.dp}_bs{args.micro_batch}",
            "wandb": {
                "enabled": True,
                "team": "collie_exp",
                "project": "adalomo",
                "group": f"llama-{args.model_size}-alpaca",
            }
        },
        "zero_optimization": {"stage": 0 if args.dp == 1 else 3},
    }
    if args.optim == "lora":
        config.peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

    model = LlamaForCausalLM.from_pretrained(model_name, config=config)
    tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.bos_token
    return model, tokenizer, config


def get_dataset(tokenizer):
    template_input = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )
    template_no_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )

    with open("alpaca_gpt4_data.json", "r", encoding="utf8") as fp:
        json_dataset = json.load(fp)
    train_dataset = [
        {
            "input": template_input.format_map(example) if example.get("input", "") != ""
            else template_no_input.format_map(example),
            "output": example['output'] + tokenizer.eos_token
        } for example in json_dataset
    ]
    train_dataset = CollieDatasetForTraining(train_dataset, tokenizer=tokenizer, max_length=2048)
    logger.info(f"Train dataset len: {len(train_dataset)}\nTrain dataset[0]: {train_dataset[0]}")
    return train_dataset


def train():
    model, tokenizer, config = get_model()
    train_dataset = get_dataset(tokenizer)
    if args.optim == "adalomo":
        optimizer = AdaLomo(
            model,
            lr=args.lr,
            loss_scale=2 ** 10,
        )
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            betas=(0.9, 0.95),
            lr=args.lr,
        )
    elif args.optim == "lora":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.95),
            lr=args.lr
        )
    else:
        raise ValueError(f"optim {args.optim} not support")

    total_step = (len(train_dataset) * config.train_epochs) // (args.micro_batch * args.dp)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_step * 0.03),
        num_training_steps=total_step
    )

    monitors = [
        StepTimeMonitor(config),
        TGSMonitor(config),
        MemoryMonitor(config),
        LossMonitor(config),
        LRMonitor(config)
    ]

    trainer = Trainer(
        model=model,
        config=config,
        loss_fn=GPTLMLoss(-100),
        optimizer=optimizer,
        train_dataset=train_dataset,
        monitors=monitors,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()
    logger.info("Save Model")
    save_path = f"./llama-{args.model_size}/{args.optim}_lr-{args.lr}_epoch-{args.train_epochs}_tp{args.tp}_dp{args.dp}_bs{args.micro_batch}"
    trainer.save_model(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp", default=1, type=int)
    parser.add_argument("--tp", default=1, type=int)
    parser.add_argument("--dp", default=8, type=int)
    parser.add_argument("--model_size", default="7b", type=str)
    parser.add_argument("--micro_batch", default=16, type=int)
    parser.add_argument("--train_epochs", default=3, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--optim", default="adalomo", type=str, choices=["adalomo", "adamw", "lora"])
    args = parser.parse_args()
    train()
