from typing import Callable, Optional, Tuple, Any, Dict

import torch
from peft import PeftModel
from torch import nn
import torch.distributed as dist
from collie import ColliePadder, GPTLMLoss, auto_param_call, BaseMetric
from collie.module import PipelineModel
from collie.controller.evaluator import Evaluator


class EvaluatorForPretraining(Evaluator):
    def __init__(self,
                 loss_fn: Callable = GPTLMLoss(),
                 collate_fn: Optional[Callable] = ColliePadder(),
                 *args,
                 **kwargs):
        self.loss_fn = loss_fn
        super().__init__(collate_fn=collate_fn, *args, **kwargs)

    @staticmethod
    @torch.no_grad()
    def eval_fn(evaluator, batch: Tuple) -> Any:
        """一次验证的基本单元

        :param evaluator: 训练器
        :param batch: 一个 batch 的数据，类型为长度为 ``Dict``，格式为：

            .. code-block::
            {
                "input_ids": torch.tensor([[1, 100, 100, 2]]),
                "labels": torch.tensor([[1, 100, 100, 2]]),
            }

        :return: 一次验证的结果，为 `Dict` 类型，该结果会被传入 `metric` 的 `update` 方法中
        """
        # concat prompt labels for p-tuning
        if evaluator.config.peft_config and evaluator.config.peft_config.peft_type in ["PROMPT_TUNING", "P_TUNING"]:
            batch_size = batch["input_ids"].shape[0]
            if "labels" in batch.keys():
                prefix_labels = torch.full((batch_size, evaluator.config.peft_config.num_virtual_tokens), -100).to(
                    batch["labels"].device)
                batch["labels"] = torch.cat((prefix_labels, batch["labels"]), dim=1)
        if evaluator.config.pp_size > 1:
            if isinstance(evaluator.engine.module, PipelineModel):
                evaluator.engine.module.forward_type = "eval"
            if isinstance(evaluator.engine.module, PeftModel) and isinstance(evaluator.engine.module.get_base_model(),
                                                                             PipelineModel):
                evaluator.engine.module.get_base_model().forward_type = "eval"
            outputs = evaluator.engine.module(**batch)
        else:
            outputs = evaluator.engine(**batch)
        loss = auto_param_call(evaluator.loss_fn, {**batch, **outputs},
                               signature_fn=evaluator.loss_fn.forward if isinstance(evaluator.loss_fn,
                                                                                    nn.Module) else evaluator.loss_fn)
        ppl = torch.exp(loss)

        # calculate acc
        pred = torch.argmax(outputs["logits"], dim=-1)
        valid_mask = (batch['labels'] != -100)
        correct = (pred == batch['labels']) & valid_mask
        correct_count = correct.float().sum().item()
        total = valid_mask.float().sum().item()
        return {
            "ppl": ppl.detach().clone().view(1, ).cuda(),
            "correct": correct_count,
            "total": total
        }


class AccMetric(BaseMetric):
    def __init__(self, gather_result: bool = False) -> None:
        super().__init__(gather_result)
        self.total = 0
        self.correct = 0

    def reset(self):
        self.total = 0
        self.correct = 0

    def get_metric(self) -> Optional[Dict]:
        return {'acc': round(self.correct / (self.total + 1e-12), 6), "total": self.total, "correct": self.correct}

    def update(self, result: Dict):
        self.total += result['total']
        self.correct += result["correct"]

    def gather(self, result):
        if self.trainer.config.dp_size > 1:
            group = self.trainer.engine.mpu.get_data_parallel_group()
            for key in result.keys():
                if key in ["total", "correct"]:
                    gather_list = [None for _ in range(self.trainer.config.dp_size)]
                    dist.all_gather_object(gather_list, result[key], group=group)
                    result[key] = sum(gather_list)
        return result
