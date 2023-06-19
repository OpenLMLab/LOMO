# LOMO: LOw-Memory Optimization

This is the implementation for [Full Parameter Fine-Tuning for Large Language Models with Limited Resources](https://arxiv.org/pdf/2306.09782.pdf).

In this work, we propose a new optimizer, **LO**w-Memory **O**ptimization (**LOMO**), which fuses the gradient computation and the parameter update in one step to reduce memory usage.
Our approach enables the full parameter fine-tuning of a 7B model on a single RTX 3090, or 
a 65B model on a single machine with 8×RTX 3090, each with 24GB memory.

LOMO is integrated with [CoLLiE](https://github.com/OpenLMLab/collie) library, which supports Collaborative Tuning of Large Language Models in an Efficient Way.

![LOMO](assets/LOMO.png)

---
## Run the code
```shell
bash run.sh
```

## Reproduce our results
We provide the sampled datasets used in our experiments [here](https://drive.google.com/drive/folders/1zV7sXvU7YHKWyS3fYV0yyi7FyTjIpEuO?usp=sharing).
Due to the limited computational resources, we reported the highest results obtained from experiments conducted with the same random seed (`42`).
We acknolwedge this limitation in our work and plan to conduct repeated experiments in the next version to address it.
Feel free to raise an issue if you have any questions.