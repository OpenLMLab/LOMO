import copy
import os
import sys

from random import sample

import torch
from torch.utils.data import Subset
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import set_seed
from dataclasses import asdict
from transformers.deepspeed import HfDeepSpeedConfig
import wandb
# os.environ['WANDB_MODE'] = 'debug'

python_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("PYTHON_PATH", python_path)
sys.path.append(python_path)
from log import print
from arguments import ModelArguments, DataArguments, MyTrainingArguments
from mydatasets import MyDataset, get_dataset_info
from mytrainer import MyInplaceZeroTrainer
from utils import DataCollatorForCauselLM, EvalDataCollatorForCauselLM


def compute_metrics(all_pred, eval_dataset, eval_prefix=None):
    golds = [ins['answer'] for ins in eval_dataset.data]
    preds = all_pred[:len(golds)]
    print(len(all_pred))
    print(all_pred[:8])
    print(all_pred[-8:])
    assert len(preds) == len(golds), f"# of predictions {len(preds)} doesn't match # of references {len(golds)}."

    acc = round(sum([int(pred == gold) for pred, gold in zip(preds, golds)]) / len(golds), 6)
    result = {'acc': acc}
    return result


def train():
    # ========== 1. logs and args ==========
    torch.set_default_dtype(torch.float16)
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    if sys.argv[-1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    model_name = model_args.model_name_or_path.split('/')[-1]
    tag_name = '_'.join([data_args.dataset_name, model_name, training_args.tag] if training_args.tag else [data_args.dataset_name, model_name])
    hparam_name = 'output'
    if training_args.optim != 'sgd':
        hparam_name += '_' + training_args.optim
    if training_args.learning_rate != 5e-4:
        hparam_name += '_lr' + str(training_args.learning_rate)
    if training_args.per_device_train_batch_size != 8:
        hparam_name += '_bs' + str(training_args.per_device_train_batch_size)
    if training_args.lr_scheduler_type != 'linear':
        hparam_name += '_' + training_args.lr_scheduler_type
    if training_args.warmup != 0:
        hparam_name += '_warmup' + str(training_args.warmup)
    if training_args.clip_grad_norm and training_args.clip_grad_norm > 0:
        hparam_name += '_clipnorm' + str(training_args.clip_grad_norm)
    if training_args.clip_grad_value and training_args.clip_grad_value > 0:
        hparam_name += '_clipgrad' + str(training_args.clip_grad_value)
    if training_args.clip_loss_value and training_args.clip_loss_value > 0:
        hparam_name += '_cliploss' + str(training_args.clip_loss_value)
    # assert training_args.clip_grad_value is None or training_args.clip_loss_value is None
    training_args.output_dir = os.path.join('outputs', tag_name, hparam_name)

    if training_args.tag == 'debug':
        os.environ['WANDB_MODE'] = 'offline'
    if training_args.local_rank in [-1, 0]:
        wandb_config = copy.deepcopy(asdict(training_args))
        wandb_config.update(asdict(model_args))
        wandb_config.update(asdict(data_args))
        wandb.init(
            project="collie",
            entity='collie_exp',
            name=tag_name if hparam_name == 'output' else '_'.join([tag_name, hparam_name.replace('output_', '')]),
            config=wandb_config
        )

    # ========== 2. Load pretrained model and tokenizer. ==========
    ds_config = training_args.deepspeed
    dschf = HfDeepSpeedConfig(ds_config)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.gradient_checkpointing = training_args.gradient_checkpointing
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
        config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        padding_side='left'
    )
    tokenizer.pad_token_id = 0

    # ========== 3. Preprocessing the datasets. ==========
    dataset_info = get_dataset_info(data_args.dataset_name)
    train_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.exemplar_split)
    eval_dataset = MyDataset(data_args, tokenizer, dataset_info, split=dataset_info.eval_split)

    # ========== 4. Initialize our Trainer. ==========
    trainer = MyInplaceZeroTrainer(
        model=model,
        training_args=training_args,
        data_collator={'train': DataCollatorForCauselLM(tokenizer, max_length=data_args.data_max_length, padding_side='left'),
                       'eval': EvalDataCollatorForCauselLM(tokenizer, max_length=data_args.data_max_length, padding_side='left')},
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    if training_args.do_train:
        trainer.train()
    else:
        trainer.eval(trainer.global_step, 0, trainer.eval_dataset, trainer.eval_dataloader, 'zero-shot')


# run with $torchrun --nproc_per_node 2 train_inplace_tensor.py config/tensor_args.yaml
if __name__ == "__main__":
    train()
