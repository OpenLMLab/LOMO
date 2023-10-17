[**English**](./README.md) | [**中文**](./README_ZH.md)

# AdaLomo: Low-memory Optimization with Adaptive Learning Rate

This is the code for [AdaLomo: Low-memory Optimization with Adaptive Learning Rate](https://arxiv.org/pdf/2310.10195.pdf).

In this work, we examined the distinctions between the LOMO and Adam optimization techniques and introduce AdaLomo, which provides an adaptive learning rate for each parameter and utilizes grouped update normalization while maintaining memory efficiency.
AdaLomo achieves results comparable to AdamW in both instruction-tuning and further pre-training with less memory footprint.

![AdaLomo](../assets/adalomo_algorithm.png)

## Dependencies
```shell
collie-lm
```

AdaLomo will be implemented at [https://github.com/OpenLMLab/collie/blob/dev/collie/optim/adalomo.py](https://github.com/OpenLMLab/collie/blob/dev/collie/optim/adalomo.py).

Code is coming soon.

[//]: # (## Run the code)

[//]: # ()
[//]: # (We provide code for fine-tuning Large Language Models &#40;LLMs&#41; using three different approaches: **LOMO**, **LoRA**, and **LoRA + LOMO**.)

[//]: # ()
[//]: # (1. For full parameter fine-tuning using LOMO, the implementation is in `src/lomo_trainer.py`, and you can run:)

[//]: # (```shell)

[//]: # (deepspeed --master_port "$port" --include localhost:"$CUDA_VISIBLE_DEVICES" src/train_lomo.py config/args_lomo.yaml)

[//]: # (```)

[//]: # ()
[//]: # (2. For LoRA and LoRA + LOMO, the implementation is in `src/lomo_lora_trainer.py`, and you can run:)

[//]: # (```shell)

[//]: # (deepspeed --master_port "$port" --include localhost:"$CUDA_VISIBLE_DEVICES" src/train_lomo_lora.py config/args_lomo_lora.yaml)

[//]: # (```)

[//]: # (In the code, we have included the `lora_only` argument in `src/arguments.py`, which controls whether to use LoRA only or LoRA + LOMO. Please note that when `lora_only` is set to `True`, the arguments related to LOMO will not work.)

[//]: # ()
[//]: # (Besides, we provide a simple `run.sh` script for convenience. You can execute the code using the following command:)

[//]: # (```shell)

[//]: # (bash run.sh)

[//]: # (```)

[//]: # ()
[//]: # (For data processing, we currently only provide the six datasets of SuperGLUE mentioned in the paper. If you wish to use new datasets, please modify the `Dataset` and `DataCollator` accordingly.)

[//]: # ()
[//]: # (For evaluation, we currently only provide the `eval_step` codes for [multiple-choice QA]&#40;https://github.com/OpenLMLab/LOMO/blob/91cc71387d0a576c000a7dc568543c4ef22401db/src/lomo_trainer.py#L259-L276&#41; and [generation]&#40;https://github.com/OpenLMLab/LOMO/blob/91cc71387d0a576c000a7dc568543c4ef22401db/src/lomo_trainer.py#L278-L297&#41; tasks. If you have other requirements, please modify the `eval_step` code in the `LOMOTrainer` or `LOMOLoRATrainer` accordingly and provide the necessary `compute_metrics` to the trainer.)

[//]: # (## Reproduce our results)

[//]: # (We provide the sampled datasets used in our experiments [here]&#40;https://drive.google.com/drive/folders/1zV7sXvU7YHKWyS3fYV0yyi7FyTjIpEuO?usp=sharing&#41;.)

[//]: # (Due to the limited computational resources, we reported the highest results obtained from experiments conducted with the same random seed &#40;`42`&#41;.)

[//]: # (We acknolwedge this limitation in our work and plan to conduct repeated experiments in the next version to address it.)

[//]: # (> Feel free to raise issues if you have any questions.)

[//]: # (## Citation)

[//]: # (```text)

[//]: # ()
[//]: # (```)