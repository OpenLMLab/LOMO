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

AdaLomo 将在 [https://github.com/OpenLMLab/collie/blob/dev/collie/optim/adalomo.py](https://github.com/OpenLMLab/collie/blob/dev/collie/optim/adalomo.py) 实现。

代码即将发布。
