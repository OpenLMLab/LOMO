[**English**](./README.md) | [**中文**](./README_ZH.md)

# LOMO: LOw-Memory Optimization

论文 [Full Parameter Fine-Tuning for Large Language Models with Limited Resources](https://arxiv.org/pdf/2306.09782.pdf) 的实现.

在这个工作中，我们提出了一个新的优化器，**LO**w-Memory **O**ptimization (**LOMO**)，它将梯度计算和参数更新融合在一步中，以减少内存使用。
我们的方法使得在单张 RTX 3090 上可以进行 7B 模型的全参数微调，或者在单个 8×RTX 3090 的机器上可以进行 65B 模型的全参数微调（RTX 3090 的内存为 24GB）。

LOMO已经集成到了 [CoLLiE](https://github.com/OpenLMLab/collie) (Collaborative Tuning of Large Language Models in an Efficient Way) 中。

---
## 依赖
```shell
torch
deepspeed
transformers
peft
wandb
```
LOMO本身只依赖 PyTorch，其他依赖用于复现我们的论文结果。
## 执行代码
```shell
bash run.sh
```
要更改模型大小、数据集或超参数，请修改`config`下的文件。

## 复现我们的结果
我们在[这里](https://drive.google.com/drive/folders/1zV7sXvU7YHKWyS3fYV0yyi7FyTjIpEuO?usp=sharing)提供了我们实验中使用的采样数据集。
由于计算资源有限，我们报告了使用相同随机种子（`42`）进行的实验中获得的最高结果。
我们在我们的工作中承认了这个限制，并计划在下一个版本中进行重复实验来解决这个问题。
如果您有任何问题，请随时提出问题。