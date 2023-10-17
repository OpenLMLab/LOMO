import argparse
import json

import torch
from transformers import get_cosine_schedule_with_warmup, LlamaTokenizer
from collie import EvalMonitor, PPLMetric, AccuracyMetric, EvaluatorForPerplexity, Callback
from collie.config import CollieConfig
from collie.controller.trainer import Trainer
from collie.module import GPTLMLoss
from collie.log import logger
from collie.optim import AdaLomo
from collie.models.llama.model import LlamaForCausalLM
from collie.utils.monitor import StepTimeMonitor, TGSMonitor, MemoryMonitor, LossMonitor, LRMonitor
from collie.data import CollieDatasetForTraining

from evaluate import EvaluatorForPretraining, AccMetric


def get_model():
    model_name = f'huggyllama/llama-{args.model_size}'

    config = CollieConfig.from_pretrained(model_name)

    config.tp_size = args.tp
    config.dp_size = args.dp
    config.pp_size = args.pp
    config.train_epochs = args.train_epochs
    config.train_micro_batch_size = args.micro_batch
    config.eval_batch_size = 2 * args.micro_batch
    config.eval_per_n_steps = 100

    config.ds_config = {
        "fp16": {"enabled": True},
        "monitor_config": {
            "enabled": True,
            "tag": f"{args.optim}-lr-{args.lr}_epoch-{config.train_epochs}_tp{args.tp}_dp{args.dp}_bs{args.micro_batch}",
            "wandb": {
                "enabled": True,
                "team": "collie_exp",
                "project": "adalomo",
                "group": f"llama-{args.model_size}-{args.domain}",
            }
        },
        "zero_optimization": {"stage": 0 if args.dp == 1 else 3},
        "zero_allow_untested_optimizer": True,
    }

    model = LlamaForCausalLM.from_pretrained(model_name, config)
    model.set_cache(False)
    tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.bos_token
    return model, tokenizer, config


def get_dataset(tokenizer):
    train_dataset = []
    eval_dataset = []
    eval_num = 2000
    if args.domain == "python":
        for i in range(10):
            logger.info(f"Loading train dataset {i}")
            data_path = f"/path_to_starcoder/train/code/python/train-000{i:02}-of-00059_train.jsonl"
            with open(data_path, "r") as f:
                for line in f:
                    train_dataset.append({'text': json.loads(line)['content']})
        data_path = f"/path_to_starcoder/train/code/python/train-00010-of-00059_train.jsonl"
        num = 0
        with open(data_path, "r") as f:
            for line in f:
                num += 1
                if num > eval_num:
                    break
                eval_dataset.append({'text': json.loads(line)['content']})
    elif args.domain == "cn":
        data_path = f"/path_to_baidu-baike/merged-1.jsonl"
        data_num = 2e6
        num = 0
        with open(data_path, "r") as f:
            for line in f:
                num += 1
                if num > data_num:
                    if num < data_num + eval_num:
                        eval_dataset.append({'text': json.loads(line)['content']})
                        continue
                    else:
                        break
                train_dataset.append({'text': json.loads(line)['content']})
    else:
        raise ValueError(f"domain {args.domain} not supported")

    train_dataset = CollieDatasetForTraining(train_dataset, tokenizer=tokenizer, max_length=2048)
    eval_dataset = CollieDatasetForTraining(eval_dataset, tokenizer=tokenizer, max_length=2048)
    return train_dataset, eval_dataset


def train():
    model, tokenizer, config = get_model()
    train_dataset, eval_dataset = get_dataset(tokenizer)
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
    else:
        raise ValueError(f"optim {args.optim} not support")

    total_step = (len(train_dataset) * config.train_epochs) // (args.micro_batch * args.dp)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_step * 0.03),
        num_training_steps=total_step
    )

    evaluator = EvaluatorForPretraining(
        model=model,
        config=config,
        dataset=eval_dataset,
        monitors=[
            EvalMonitor(config)
        ],
        metrics={
            'ppl': PPLMetric(gather_result=True),
            'acc': AccMetric(gather_result=True),
        }
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
        evaluators=[evaluator],
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp", default=1, type=int)
    parser.add_argument("--tp", default=1, type=int)
    parser.add_argument("--dp", default=8, type=int)
    parser.add_argument("--model_size", default="7b", type=str)
    parser.add_argument("--micro_batch", default=16, type=int)
    parser.add_argument("--train_epochs", default=1, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)

    parser.add_argument("--optim", default="adalomo", type=str, choices=["adalomo", "adamw"])
    parser.add_argument("--domain", default="python", type=str)

    args = parser.parse_args()
    train()
