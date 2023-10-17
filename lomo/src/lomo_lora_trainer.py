import os
import sys
import operator
from collections import OrderedDict
from itertools import chain
from pathlib import Path
import shutil

import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DistributedSampler, DataLoader
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler, SequentialDistributedSampler, nested_numpify
from transformers.trainer_utils import has_length, seed_worker
from transformers import GenerationConfig
from transformers.optimization import AdamW, get_scheduler

try:
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.accelerator import get_accelerator
except:
    pass

from src.utils import LearningRateScheduler, WandbLogger, get_loss
from log import print
from peft import get_peft_model_state_dict


class LOMOLoRATrainer:
    def __init__(
            self,
            model,
            training_args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            compute_metrics,
            optimizers=None,
    ):
        self.training_args = training_args
        if self.training_args.world_size > 1:
            raise NotImplementedError("Distributed training for LOMO+LoRA is not supported yet.")
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.wandb = WandbLogger(training_args)
        self.allow_print = self.training_args.local_rank in [0, -1]
        if self.training_args.do_eval:
            self.metrics = {}
            self.compute_metrics = compute_metrics

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
        # for grad norm
        self.gather_norm = False
        self.grad_norms = []
        self.clip_coef = None

        hf_optimizer = None
        hf_lr_scheduler = None
        if self.training_args.do_train:
            hf_optimizer = AdamW(optimizers['model_parameters'], lr=training_args.hf_learning_rate,
                                 weight_decay=training_args.hf_weight_decay)
            hf_lr_scheduler = get_scheduler(training_args.hf_lr_scheduler_type,
                                            optimizer=hf_optimizer,
                                            num_warmup_steps=training_args.hf_warmup * self.n_steps if training_args.hf_warmup < 1 else training_args.hf_warmup,
                                            num_training_steps=self.n_steps)

        if 'deepspeed' not in sys.modules:
            raise ModuleNotFoundError(
                "Detected DeepSpeed is not installed. See https://github.com/microsoft/DeepSpeed")

        # Initialize deepspeed engine
        self.model, self.peft_optimizer, _, self.peft_lr_scheduler = deepspeed.initialize(
            config=training_args.deepspeed,
            model=model,
            model_parameters=optimizers['model_parameters'] if self.training_args.do_train else None,
            optimizer=hf_optimizer,
            lr_scheduler=hf_lr_scheduler
        )

        if not self.training_args.lora_only:
            # register inplace grad hook
            self.grad_func = self.inplace_grad()
            for n, p in model.named_parameters():
                if "lora_" not in n and p.requires_grad:
                    p.register_hook(self.grad_func)

            # self.dummy_optimizer = DeepSpeedZeRoOffload(
            #     self.model.module,
            #     timers=self.model.timers if self.model.wall_clock_breakdown() else None,
            #     ds_config=self.model.config,
            #     overlap_comm=self.model.zero_overlap_comm(),
            #     prefetch_bucket_size=self.model.zero_prefetch_bucket_size(),
            #     max_reuse_distance=self.model.zero_max_reuse_distance(),
            #     max_live_parameters=self.model.zero_max_live_parameters(),
            #     param_persistence_threshold=self.model.zero_param_persistence_threshold(),
            #     model_persistence_threshold=self.model.zero_model_persistence_threshold(),
            #     offload_param_config=self.model.zero_offload_param(),
            #     mpu=self.model.mpu
            # )

        get_accelerator().empty_cache()

    def inplace_grad(self):
        # An approximation of in-place grad update under zero3 of deepspeed
        def func(x):
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if "lora_" in n:
                        continue

                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG, async_op=False)
                        if self.gather_norm:
                            grad_fp32 = p.grad.detach().clone().to(torch.float32)
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                            p.grad = None
                        else:
                            one_dim_grad = p.grad.view(-1)
                            partition_size = p.ds_tensor.numel()
                            start = partition_size * self.training_args.local_rank
                            end = start + partition_size

                            if end > p.grad.numel():
                                partitioned_grad = one_dim_grad.narrow(0, start, p.grad.numel() - start)
                                # partitioned_grad = torch.cat([partitioned_grad, torch.zeros(end - p.grad.numel()).cuda()])
                                partitioned_p = p.ds_tensor.narrow(0, 0, p.grad.numel() - start)
                                partitioned_grad_fp32 = partitioned_grad.detach().clone().to(torch.float32)
                                partitioned_p_fp32 = partitioned_p.detach().clone().to(torch.float32)
                                if self.training_args.clip_grad_value is not None:
                                    # Gradients are modified in-place.
                                    partitioned_grad_fp32.clamp_(min=-self.training_args.clip_grad_value,
                                                                 max=self.training_args.clip_grad_value)
                                if self.training_args.clip_grad_norm is not None and self.training_args.clip_grad_norm > 0 and self.clip_coef is not None:
                                    partitioned_grad_fp32.mul_(self.clip_coef)
                                partitioned_p_fp32.add_(partitioned_grad_fp32, alpha=-self.lr)
                                partitioned_p.copy_(partitioned_p_fp32)
                            else:
                                partitioned_grad = one_dim_grad.narrow(0, start, partition_size)
                                partitioned_grad_fp32 = partitioned_grad.detach().clone().to(torch.float32)
                                if self.training_args.clip_grad_value is not None:
                                    # Gradients are modified in-place.
                                    partitioned_grad_fp32.clamp_(min=-self.training_args.clip_grad_value,
                                                                 max=self.training_args.clip_grad_value)
                                if self.training_args.clip_grad_norm is not None and self.training_args.clip_grad_norm > 0 and self.clip_coef is not None:
                                    partitioned_grad_fp32.mul_(self.clip_coef)
                                ds_tensor_fp32 = p.ds_tensor.detach().clone().to(torch.float32)
                                ds_tensor_fp32.add_(partitioned_grad_fp32, alpha=-self.lr)
                                p.ds_tensor.copy_(ds_tensor_fp32)
                            p.grad = None
            return x

        return func

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
                    if self.training_args.lora_only:
                        self.global_step = self.num_steps_per_epoch * epoch + step

                        loss = self.model.backward(loss)
                        self.model.step()  # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                    else:
                        self.global_step = self.num_steps_per_epoch * epoch + step
                        self.lr = self.lr_scheduler.step(self.global_step)

                        if self.training_args.clip_grad_norm is not None and self.training_args.clip_grad_norm > 0:
                            self.gather_norm = True
                            self.grad_norms = []

                            self.model.backward(loss)
                            # update the last one since the hook function will not be called for the last parameter
                            self.grad_func(0)
                            # self.model.optimizer._get_param_coordinator(training=True).reset_step()
                            # self.dummy_optimizer.get_param_coordinator(training=True).reset_step()

                            with torch.no_grad():
                                # The norm is computed over all gradients together, as if they were
                                # concatenated into a single vector. Gradients are modified in-place.
                                self.grad_norms = torch.stack(self.grad_norms)
                                device = torch.device(f"cuda:{self.training_args.local_rank}")
                                all_grad_norms = torch.zeros(self.training_args.world_size * self.grad_norms.shape[0],
                                                             dtype=self.grad_norms.dtype, device=device)
                                torch.distributed.all_gather_into_tensor(all_grad_norms, self.grad_norms)

                                total_norm = torch.norm(all_grad_norms, 2.0)
                                self.clip_coef = float(self.training_args.clip_grad_norm) / (total_norm + 1e-6)
                                self.clip_coef = torch.clamp(self.clip_coef, max=1.0)
                            self.gather_norm = False

                            # 第二次forward
                            outs = self.model(
                                input_ids=batch['input_ids'].cuda(),
                                attention_mask=batch['attention_mask'].cuda(),
                            )
                            loss = get_loss(outs.logits, batch['labels'], self.training_args.clip_loss_value)

                        # update peft params
                        loss = self.model.backward(loss)
                        self.grad_func(0)
                        self.model.step()  # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps

                    tqb.set_postfix({'loss': loss.item()})
                    if self.allow_print:
                        self.wandb.log(
                            {
                                'train/loss': loss.item(),
                                'train/learning_rate': self.lr,
                                'train/hf_learning_rate': self.model.get_lr()[0],
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
                    if self.training_args.predict_with_generate:
                        pred = self.generate_step(batch)
                    else:
                        pred = self.eval_step(batch)
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

        state_dict = OrderedDict() if torch.distributed.get_rank() == 0 else None
        shared_params = {}

        # Prepare for checkpoint save by ensuring all parameters are partitioned
        self.model.optimizer.partition_all_parameters()

        for name, param in self.model.module.named_parameters():
            with deepspeed.zero.GatheredParameters(param):
                if torch.distributed.get_rank() == 0:
                    # can't rely on param.data_ptr() as it will be reused as weights gets
                    # gathered and reduced, but param.ds_id is unique across all zero weights
                    # (and shared params will have the same param.ds_id)
                    if param.ds_id in shared_params:
                        # shared weights
                        state_dict[name] = state_dict[shared_params[param.ds_id]]
                    else:
                        state_dict[name] = param.detach().cpu()
                        shared_params[param.ds_id] = name

        if len(self.model.optimizer.persistent_parameters) > 0:
            self.model.optimizer.persistent_parameters[0].all_gather(self.model.optimizer.persistent_parameters)

        if torch.distributed.get_rank() == 0:
            self.model.module.config.save_pretrained(output_dir)
            torch.save(state_dict, os.path.join(output_dir, 'pytorch_model.bin'))
            print(f"Save model to {output_dir}")

            # save lora
            adapter_output_dir = os.path.join(output_dir, 'adapter_model')
            self.model.module.peft_config['default'].save_pretrained(adapter_output_dir)
            # if state dict is not what you expected, you can use the following code to get the state dict
            # engine_state_dict = self.model._zero3_consolidated_16bit_state_dict()
            lora_state_dict = get_peft_model_state_dict(self.model.module, state_dict)
            torch.save(lora_state_dict, os.path.join(adapter_output_dir, "adapter_model.bin"))
            print(f"Save adapter model at {adapter_output_dir}")

        torch.distributed.barrier()
