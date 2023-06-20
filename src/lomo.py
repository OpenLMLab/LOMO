import os
import torch
from torch.optim import Optimizer
import torch.distributed as dist

from src.utils import DynamicLossScaler


class LOMO(Optimizer):
    """
    一个自定义的优化器类LOMO，用于在分布式训练中的梯度更新。

    该类实现两个梯度更新函数 :meth:`fuse_update` 和 :meth:`fuse_update_zero3`，分别用于非ZeRO和ZeRO模式下的梯度更新。

    :param model: 待优化的模型
    :param lr: 学习率，默认值为1e-3
    :param clip_grad_norm: 梯度裁剪的范数阈值

        .. note::

            clip_grad_norm须为正数

    :param clip_grad_value: 梯度裁剪的值域阈值
    """

    def __init__(self, model, lr=1e-3, clip_grad_norm=None, clip_grad_value=None):
        self.model = model
        self.lr = lr
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = dist.get_world_size()
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        # for grad norm
        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError(f"clip_grad_norm should be positive, got {self.clip_grad_norm}.")
        self.gather_norm = False
        self.grad_norms = []
        self.clip_coef = None

        # check if zero3 is enabled
        p0 = list(self.model.parameters())[0]
        if hasattr(p0, 'ds_tensor'):  # zero3 is enabled
            self.grad_func = self.fuse_update_zero3()
        else:
            self.grad_func = self.fuse_update()
        # check if fp16 is enabled
        if p0.dtype == torch.float16:
            self.loss_scaler = DynamicLossScaler(
                init_scale=2 ** 16,
            )  # TODO: add args
        else:
            self.loss_scaler = None

        # register hook function, which will be called through the backward process
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.register_hook(self.grad_func)
        defaults = dict(lr=lr, clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value)
        super(LOMO, self).__init__(self.model.parameters(), defaults)

    def fuse_update(self):
        """
        在非ZeRO模式下更新模型参数的梯度。

        :return: func，一个闭包函数，用于更新模型参数的梯度
        """

        def func(x):
            """
            闭包函数，用于更新模型参数的梯度。
            """
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        if self.loss_scaler and self.loss_scaler.has_overflow_serial or self.loss_scaler._has_inf_or_nan(p.grad):
                            # if the overflow is detected, drop the gradient
                            p.grad = None
                            self.loss_scaler.has_overflow_serial = True
                            break
                        if self.gather_norm:
                            # we adopt two backward pass for gradient norm compuation and parameter update, respectively.
                            grad_fp32 = p.grad.detach().clone().to(torch.float32)
                            if self.loss_scaler:
                                grad_fp32.div_(self.loss_scaler.loss_scale)
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                            p.grad = None
                        else:
                            grad_fp32 = p.grad.detach().clone().to(torch.float32)
                            p.grad = None
                            if self.loss_scaler:
                                grad_fp32.div_(self.loss_scaler.loss_scale)
                            if self.clip_grad_value is not None and self.clip_grad_value > 0:
                                # Clipping gradients by their value
                                grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)
                            if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                                # Normalize the gradient according to its norm (computed in another pass)
                                grad_fp32.mul_(self.clip_coef)
                            p_fp32 = p.data.detach().clone().to(torch.float32)
                            p_fp32.add_(grad_fp32, alpha=-self.lr)
                            p.data.copy_(p_fp32)

            return x

        return func

    def fuse_update_zero3(self):
        """
        在ZeRO模式下更新模型参数的梯度。

        :return: func，一个闭包函数，用于更新模型参数的梯度。
        """
        def func(x):
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG, async_op=False)
                        if self.loss_scaler and self.loss_scaler.has_overflow_serial or self.loss_scaler._has_inf_or_nan(p.grad):
                            # if the overflow is detected, drop the gradient
                            p.grad = None
                            self.loss_scaler.has_overflow_serial = True
                            break

                        if self.gather_norm:
                            # we adopt two backward pass for gradient norm compuation and parameter update, respectively.
                            grad_fp32 = p.grad.detach().clone().to(torch.float32)
                            if self.loss_scaler:
                                grad_fp32.div_(self.loss_scaler.loss_scale)
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                            p.grad = None
                        else:
                            one_dim_grad = p.grad.view(-1)
                            partition_size = p.ds_tensor.numel()
                            start = partition_size * self.local_rank
                            end = start + partition_size
                            if end > p.grad.numel():
                                partitioned_grad = one_dim_grad.narrow(0, start, p.grad.numel() - start)
                                # partitioned_grad = torch.cat([partitioned_grad, torch.zeros(end - p.grad.numel()).cuda()])
                                partitioned_p = p.ds_tensor.narrow(0, 0, p.grad.numel() - start)
                                partitioned_grad_fp32 = partitioned_grad.detach().clone().to(torch.float32)
                                p.grad = None
                                if self.loss_scaler:
                                    partitioned_grad_fp32.div_(self.loss_scaler.loss_scale)
                                partitioned_p_fp32 = partitioned_p.detach().clone().to(torch.float32)
                                if self.clip_grad_value is not None:
                                    # Clipping gradients by their value
                                    partitioned_grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)
                                if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                                    # Normalize the gradient according to its norm (computed in another pass)
                                    partitioned_grad_fp32.mul_(self.clip_coef)
                                partitioned_p_fp32.add_(partitioned_grad_fp32, alpha=-self.lr)
                                partitioned_p.copy_(partitioned_p_fp32)
                            else:
                                partitioned_grad = one_dim_grad.narrow(0, start, partition_size)
                                partitioned_grad_fp32 = partitioned_grad.detach().clone().to(torch.float32)
                                p.grad = None
                                if self.loss_scaler:
                                    partitioned_grad_fp32.div_(self.loss_scaler.loss_scale)
                                if self.clip_grad_value is not None:
                                    # Clipping gradients by their value
                                    partitioned_grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)
                                if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                                    # Normalize the gradient according to its norm (computed in another pass)
                                    partitioned_grad_fp32.mul_(self.clip_coef)
                                ds_tensor_fp32 = p.ds_tensor.detach().clone().to(torch.float32)
                                ds_tensor_fp32.add_(partitioned_grad_fp32, alpha=-self.lr)
                                p.ds_tensor.copy_(ds_tensor_fp32)
                            p.grad = None
            return x

        return func

    def fused_backward(self, loss, lr):
        """
        执行一步反向传播并更新模型的梯度。

        :param loss: 模型的loss值
        :param lr: 学习率
        """
        self.lr = lr
        # Users need call grad_norm themselves and then call backward_step
        if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is None:
            raise ValueError(
                "clip_grad_norm is not None, but clip_coef is None. "
                "Please call optimizer.grad_norm() before backward_step."
            )
        if self.loss_scaler:
            loss = loss * self.loss_scaler.loss_scale
        loss.backward()
        # update the last parameter since the last parameter in the computaiton graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything. 
        self.grad_func(0)

    def grad_norm(self, loss):
        """
        计算梯度的范数。

        :param loss: 模型的loss值
        """
        self.gather_norm = True
        self.grad_norms = []
        if self.loss_scaler:
            self.loss_scaler.has_overflow_serial = False
            loss = loss * self.loss_scaler.loss_scale
        loss.backward(retain_graph=True)
        # update the last parameter since the last parameter in the computaiton graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything. 
        self.grad_func(0)

        if self.loss_scaler and self.loss_scaler.has_overflow_serial:
            self.loss_scaler.update_scale(overflow=True)
            with torch.no_grad():  # clear gradients
                for n, p in self.model.named_parameters():
                    p.grad = None
            return


        with torch.no_grad():
            # The norm is computed over all gradients together, as if they were
            # concatenated into a single vector. Gradients are modified in-place.
            self.grad_norms = torch.stack(self.grad_norms)

            total_norm = torch.norm(self.grad_norms, 2.0)
            self.clip_coef = float(self.clip_grad_norm) / (total_norm + 1e-6)
            self.clip_coef = torch.clamp(self.clip_coef, max=1.0)
        self.gather_norm = False
