import os
import sys
import operator
from collections import OrderedDict
from itertools import chain
from pathlib import Path
import shutil

import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DistributedSampler, DataLoader
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler, SequentialDistributedSampler, nested_numpify
from transformers.trainer_utils import has_length, seed_worker
from transformers import GenerationConfig

try:
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.accelerator import get_accelerator
except:
    pass

from src.utils import LearningRateScheduler, WandbLogger, DynamicLossScaler, get_loss
from src.lomo import LOMO
from log import print


class LOMOTrainer:
    def __init__(
            self,
            model,
            training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
    ):
        self.training_args = training_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.wandb = WandbLogger(training_args)
        self.allow_print = self.training_args.local_rank in [-1, 0]
        if self.training_args.do_eval:
            self.metrics = {}
            self.compute_metrics = compute_metrics

        if 'deepspeed' not in sys.modules:
            raise ModuleNotFoundError(
                "Detected DeepSpeed is not installed. See https://github.com/microsoft/DeepSpeed")

        # Initialize deepspeed engine
        self.model, _, _, _ = deepspeed.initialize(
            config=training_args.deepspeed,
            model=model,
        )

        # get train_dataloader and eval_dataloader
        if isinstance(data_collator, dict):
            assert 'train' in data_collator and 'eval' in data_collator, "data_collator should be a dict with keys 'train' and 'eval'."
            self.train_data_collator = data_collator['train']
            if self.training_args.do_eval:
                self.eval_data_collator = data_collator['eval']
        else:
            self.train_data_collator = self.eval_data_collator = data_collator
        self.train_dataloader = self.get_train_dataloader()
        if self.training_args.do_eval:
            if isinstance(self.eval_dataset, dict):
                self.eval_dataloader = {}
                for prefix in self.eval_dataset.keys():
                    self.eval_dataloader[prefix] = self.get_eval_dataloader(self.eval_dataset[prefix])
            else:
                self.eval_dataloader = self.get_eval_dataloader()

        # setup learning rate
        self.num_steps_per_epoch = len(self.train_dataloader)
        self.global_step = 1
        self.n_steps = self.num_steps_per_epoch * self.training_args.num_train_epochs
        self.lr_scheduler = LearningRateScheduler(learning_rate=self.training_args.learning_rate,
                                                  warmup=self.training_args.warmup,
                                                  schedule=self.training_args.lr_scheduler_type,
                                                  n_steps=self.n_steps)
        self.lr = 0

        self.optimizer = LOMO(model, self.lr, training_args.clip_grad_norm, training_args.clip_grad_value)

        get_accelerator().empty_cache()

    def train(self):
        for epoch in range(self.training_args.num_train_epochs):
            print(f"***** Running Training *****")
            print(f"  Num examples: {len(self.train_dataset)}")
            print(f"  Num Epochs: {self.training_args.num_train_epochs}")
            print(f"  Current Epoch: {epoch}")
            print(f"  Batch Size: {self.training_args.per_device_train_batch_size}")
            if self.allow_print:
                self.wandb.log({'train/epoch': epoch}, step=self.global_step)
            self.train_dataloader.sampler.set_epoch(epoch)

            with tqdm.tqdm(self.train_dataloader, disable=not self.allow_print) as tqb:
                for step, batch in enumerate(tqb, start=1):
                    self.model.train()
                    outs = self.model(
                        input_ids=batch['input_ids'].cuda(),
                        attention_mask=batch['attention_mask'].cuda(),
                    )
                    loss = get_loss(outs.logits, batch['labels'], self.training_args.clip_loss_value)

                    # update the learning rate
                    self.global_step = self.num_steps_per_epoch * epoch + step
                    self.lr = self.lr_scheduler.step(self.global_step)
                    if self.training_args.clip_grad_norm is not None and self.training_args.clip_grad_norm > 0:
                        self.optimizer.grad_norm(loss)
                        # self.gather_norm = True
                        # self.grad_norms = []
                        # self.loss_scaler.has_overflow_serial = False
                        # scaled_loss = loss * self.loss_scaler.loss_scale
                        #
                        # scaled_loss.backward()
                        # # update the last one since the hook function will not be called for the last parameter
                        # self.grad_func(0)

                        if self.optimizer.loss_scaler and self.optimizer.loss_scaler.has_overflow_serial:
                            print(f"Gradient overflow, skipping step {self.global_step}")
                            # self.loss_scaler.update_scale(overflow=True)
                            # with torch.no_grad():
                            #     for n, p in self.model.named_parameters():
                            #         p.grad = None
                            self.model.optimizer.get_param_coordinator(training=True).reset_step()
                            tqb.set_postfix({'loss': loss.item()})
                            if self.allow_print:
                                self.wandb.log(
                                    {
                                        'train/loss': loss.item(),
                                        'train/learning_rate': self.lr,
                                        'train/global_step': self.global_step,
                                    },
                                    step=self.global_step
                                )
                            continue

                        # with torch.no_grad():
                        #     # The norm is computed over all gradients together, as if they were
                        #     # concatenated into a single vector. Gradients are modified in-place.
                        #     self.grad_norms = torch.stack(self.grad_norms)
                        #     # device = torch.device(f"cuda:{self.training_args.local_rank}")
                        #     # all_grad_norms = torch.zeros(self.training_args.world_size * self.grad_norms.shape[0], dtype=self.grad_norms.dtype, device=device)
                        #     # torch.distributed.all_gather_into_tensor(all_grad_norms, self.grad_norms)
                        #
                        #     # total_norm = torch.norm(all_grad_norms, 2.0) / self.training_args.world_size
                        #     total_norm = torch.norm(self.grad_norms, 2.0)
                        #     self.clip_coef = float(self.training_args.clip_grad_norm) / (total_norm + 1e-6)
                        #     self.clip_coef = torch.clamp(self.clip_coef, max=1.0)
                        # self.gather_norm = False
                        else:
                            self.model.optimizer.get_param_coordinator(training=True).reset_step()
                        # 第二次forward
                        outs = self.model(
                            input_ids=batch['input_ids'].cuda(),
                            attention_mask=batch['attention_mask'].cuda(),
                        )
                        loss = get_loss(outs.logits, batch['labels'], self.training_args.clip_loss_value)

                    # scaled_loss = loss * self.loss_scaler.loss_scale
                    #
                    # scaled_loss.backward()
                    # # update the last one since the hook function will not be called for the last parameter
                    # self.grad_func(0)
                    # self.loss_scaler.update_scale(overflow=False)
                    self.optimizer.fused_backward(loss, self.lr)
                    self.model.optimizer.get_param_coordinator(training=True).reset_step()

                    tqb.set_postfix({'loss': loss.item()})
                    if self.allow_print:
                        self.wandb.log(
                            {
                                'train/loss': loss.item(),
                                'train/learning_rate': self.lr,
                                'train/global_step': self.global_step,
                            },
                            step=self.global_step
                        )

                    if self.training_args.save_strategy == 'steps' and self.global_step % self.training_args.save_steps == 0:
                        self.save_model(self.global_step)

                    if self.training_args.do_eval and self.training_args.evaluation_strategy == 'steps' and \
                            self.global_step % self.training_args.eval_steps == 0:
                        if isinstance(self.eval_dataset, dict):
                            for prefix in self.eval_dataset.keys():
                                assert prefix in self.eval_dataloader.keys(), "eval_dataset and eval_dataloader should have the same keys."
                                self.eval(self.global_step, epoch, self.eval_dataset[prefix],
                                          self.eval_dataloader[prefix], prefix)
                        else:
                            self.eval(self.global_step, epoch, self.eval_dataset, self.eval_dataloader, 'eval')

            if self.training_args.save_strategy == 'epoch':
                self.save_model(epoch)

            if self.training_args.do_eval and self.training_args.evaluation_strategy == 'epoch':
                if isinstance(self.eval_dataset, dict):
                    for prefix in self.eval_dataset.keys():
                        assert prefix in self.eval_dataloader.keys(), "eval_dataset and eval_dataloader should have the same keys."
                        self.eval(self.global_step, epoch, self.eval_dataset[prefix], self.eval_dataloader[prefix],
                                  prefix)
                else:
                    self.eval(self.global_step, epoch, self.eval_dataset, self.eval_dataloader, 'eval')

    def eval(
            self,
            step: int,
            epoch: int,
            dataset: torch.utils.data.Dataset,
            dataloader: DataLoader,
            eval_prefix: str
    ):
        r"""
                Shared by both eval(validation) and predict(test).
                This method will be called by the trainer to evaluate the model.
                """
        print(f"***** Running {eval_prefix} *****")
        print(f"  Num examples: {len(dataset)}")
        print(f"  Current Epoch: {epoch}")
        print(f"  Batch size: {self.training_args.per_device_eval_batch_size}")

        with tqdm.tqdm(dataloader, disable=not self.allow_print) as tqb:
            all_preds = None
            self.model.eval()
            for batch in tqb:
                with torch.no_grad():
                    pred = self.eval_step(batch)  # change to generate_step() for generation tasks
                    all_preds = pred if all_preds is None else all_preds + pred

            all_preds_gather = [None for _ in range(self.training_args.world_size)]
            torch.distributed.all_gather_object(all_preds_gather, all_preds)
            all_pred_merged = list(chain(*all_preds_gather))

            result = self.compute_metrics(all_pred_merged, dataset, eval_prefix)
            result = {f"{eval_prefix}/{k}": v for k, v in result.items()}
            prefix_metric_for_best_model = f'{eval_prefix}/{self.training_args.metric_for_best_model}'
            result_value = result[prefix_metric_for_best_model]

            if self.allow_print:
                print(f'epoch: {epoch}, step: {step}, {self.training_args.metric_for_best_model}: {result_value}')
                self.wandb.log(result, step=step)

                if self.is_better(result, prefix_metric_for_best_model):
                    self.wandb.set_summary(f'{eval_prefix}/best_{self.training_args.metric_for_best_model}', result_value)
                    self.wandb.set_summary(f'{eval_prefix}/best_epoch', epoch)
                    self.wandb.set_summary(f'{eval_prefix}/best_step', step)
                    self.metrics[prefix_metric_for_best_model] = result_value

    def eval_step(self, batch):
        """
        used for classification or multi-choice qa tasks in eval()
        """
        outs = self.model(batch['input_ids'].cuda(), batch['attention_mask'].cuda())
        # Shift so that tokens < n predict n
        shift_logits = outs.logits[..., :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].cuda().contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                        shift_labels.view(-1)).view_as(shift_labels)
        loss = loss.mean(dim=1)
        group_loss = loss.split(batch['split_size'])
        preds = torch.stack([torch.argmin(l) for l in group_loss], dim=0)

        preds = nested_numpify(preds)
        return preds.tolist()

    def generate_step(self, batch):
        """
        used for generation tasks in eval()
        """
        self.model.eval()
        generation_config = GenerationConfig(max_length=self.training_args.max_length,
                                             max_new_tokens=self.training_args.max_new_tokens,
                                             do_sample=self.training_args.do_sample,
                                             temperature=self.training_args.temperature,
                                             top_k=self.training_args.top_k,
                                             top_p=self.training_args.top_p,
                                             typical_p=self.training_args.typical_p,
                                             repetition_penalty=self.training_args.repetition_penalty, )
        logits = self.model.generate(
            inputs=batch['input_ids'].cuda(),
            generation_config=generation_config
        )
        predictions = logits.detach().cpu().numpy()
        pred_texts = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        return pred_texts

    def is_better(self, result_dict, key):
        """
        判断 ``result`` 是否更好。

        :param result:
        """
        op = operator.gt if self.training_args.greater_is_better else operator.lt
        return (
                key not in self.metrics or op(result_dict[key], self.metrics[key])
        )

    def get_train_sampler(self):
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
        # `self.training_args.seed`) if data_seed isn't provided.
        # Further on in this method, we default to `self.training_args.seed` instead.
        seed = self.training_args.data_seed if self.training_args.data_seed is not None else self.training_args.seed

        if self.training_args.group_by_length:
            return DistributedLengthGroupedSampler(
                self.training_args.per_device_train_batch_size * self.training_args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                num_replicas=self.training_args.world_size,
                rank=self.training_args.local_rank,
                lengths=None,
                model_input_name="input_ids",
                seed=seed,
            )
        else:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.training_args.world_size,
                rank=self.training_args.local_rank,
                seed=seed
            )

    def get_train_dataloader(self):
        """
            Returns the training [`~torch.utils.data.DataLoader`].
            Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
            training if necessary) otherwise.
            Subclass and override this method if you want to inject some custom behavior.
            """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        data_collator = self.train_data_collator
        train_sampler = self.get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.training_args.dataloader_drop_last,
            num_workers=self.training_args.dataloader_num_workers,
            pin_memory=self.training_args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def get_eval_sampler(self, eval_dataset):
        return SequentialDistributedSampler(
            eval_dataset,
            num_replicas=self.training_args.world_size,
            rank=self.training_args.local_rank,
            # batch_size=self.training_args.per_device_eval_batch_size
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """
            Returns the evaluation [`~torch.utils.data.DataLoader`].

            Subclass and override this method if you want to inject some custom behavior.

            Args:
                eval_dataset (`torch.utils.data.Dataset`, *optional*):
                    If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                    by the `model.forward()` method are automatically removed. It must implement `__len__`.
            """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.eval_data_collator

        eval_sampler = self.get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.training_args.dataloader_drop_last,
            num_workers=self.training_args.dataloader_num_workers,
            pin_memory=self.training_args.dataloader_pin_memory,
        )

    def save_model(self, index):
        if self.training_args.local_rank in [-1, 0]:
            checkpoint_dir = sorted(Path(self.training_args.output_dir).glob("checkpoint-*"))
            if len(checkpoint_dir) >= self.training_args.save_total_limit:
                shutil.rmtree(checkpoint_dir[0], ignore_errors=True)
        torch.distributed.barrier()

        output_dir = os.path.join(self.training_args.output_dir, f"checkpoint-{index}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        state_dict = OrderedDict()
        for n, p in self.model.module.named_parameters():
            state_dict[n] = (p.ds_tensor.detach().cpu(), p.ds_numel, p.ds_shape)
        # save model shards
        if self.training_args.local_rank not in [-1, 0]:
            with open(os.path.join(output_dir, f'pytorch_model-{self.training_args.local_rank}.bin'), 'wb') as f:
                torch.save(state_dict, f)
        torch.distributed.barrier()
        # merge model shards
        if self.training_args.local_rank in [-1, 0]:
            # save config
            self.model.module.config.save_pretrained(output_dir)
            for rank in range(1, self.training_args.world_size):
                with open(os.path.join(output_dir, f'pytorch_model-{rank}.bin'), 'rb') as f:
                    state_dict_rank = torch.load(f)
                    for n in state_dict_rank:
                        state_dict[n] = (
                            torch.cat([state_dict[n][0], state_dict_rank[n][0]], dim=0),
                            state_dict[n][1],
                            state_dict[n][2]
                        )
                # remove shard files
                os.remove(os.path.join(output_dir, f'pytorch_model-{rank}.bin'))
            # reshape to original shape
            for n in state_dict:
                numel = state_dict[n][1]
                shape = state_dict[n][2]
                state_dict[n] = state_dict[n][0][:numel].view(shape)

            # save inv_freq for llama
            if self.model.module.config.model_type == "llama":
                num_layers = self.model.module.config.num_hidden_layers
                head_dim = self.model.module.config.hidden_size // self.model.module.config.num_attention_heads
                base = 10000.0
                inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
                for layer in range(num_layers):
                    state_dict[f'model.layers.{layer}.self_attn.rotary_emb.inv_freq'] = inv_freq


            with open(os.path.join(output_dir, f'pytorch_model.bin'), 'wb') as f:
                torch.save(state_dict, f)
                print(f"Save model to {output_dir}.")
        torch.distributed.barrier()
