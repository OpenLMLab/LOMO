[**English**](./README.md) | [**中文**](./README_ZH.md)

# AdaLomo: Low-memory Optimization with Adaptive Learning Rate

这是 [AdaLomo: Low-memory Optimization with Adaptive Learning Rate](https://arxiv.org/pdf/2310.10195.pdf) 的代码。

在这个工作中，我们研究了 LOMO 和 Adam 优化技术之间的区别，并提出了 AdaLomo，它为每个参数提供自适应学习率，并利用分组更新归一化来保持内存效率。
AdaLomo 在指令微调和进一步预训练中的结果与 AdamW 相当，但内存占用更少。

![AdaLomo](../assets/adalomo_algorithm.png)

## 依赖
```shell
collie-lm
```

AdaLomo 在 [https://github.com/OpenLMLab/collie/blob/dev/collie/optim/adalomo.py](https://github.com/OpenLMLab/collie/blob/dev/collie/optim/adalomo.py) 中实现。

## 指令微调
我们使用 Alpaca-GPT4 作为我们的训练数据集，该数据集可在 https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json 获取.

### 下载数据集
```shell
cd instruction-tuning
wget https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json
```

### 训练
```shell
torchrun --nproc_per_node=8 train.py --optim adalomo --model_size 7b
```

### 评估
评估基于 opencompass。以下是快速安装步骤。
```shell
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/KaiLv69/opencompass opencompass
cd opencompass
pip install -e .
```
以下是评估步骤。
```shell
python run.py configs/eval_collie.py -r
```
`-r` 用于恢复之前的评估过程。

您可以参考 `opencompass/configs/eval_collie.py` 了解更多细节。

## 继续预训练

### 获取数据集

下载 StarCoder 的 python 子集，并在 `further-pretraining/train.py` 的 `get_dataset()` 中设置路径。

### 训练
```shell
torchrun --nproc_per_node=8 train.py --optim adalomo --model_size 7b
```
