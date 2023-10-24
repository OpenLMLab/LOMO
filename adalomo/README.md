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

AdaLomo is implemented at [https://github.com/OpenLMLab/collie/blob/dev/collie/optim/adalomo.py](https://github.com/OpenLMLab/collie/blob/dev/collie/optim/adalomo.py).

## Instruction-tuning
We use Alpaca-GPT4 as our training dataset, which is available at https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json.

### Download the dataset
```shell
cd instruction-tuning
wget https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json
```

### Training
```shell
torchrun --nproc_per_node=8 train.py --optim adalomo --model_size 7b
```

### Evaluation
The evaluation is based on opencompass. Below are the steps for quick installation.
```shell
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/KaiLv69/opencompass opencompass
cd opencompass
pip install -e .
```
Below are the steps for evaluation.
```shell
python run.py configs/eval_collie.py -r
```
`-r` is for resuming the previous evaluation process.

You may refer to `opencompass/configs/eval_collie.py` for more details.

## Further pre-training

### Get dataset

Download python subset of StarCoder and set the path in the `get_dataset()` in `further-pretraining/train.py`. 

### Training
```shell
torchrun --nproc_per_node=8 train.py --optim adalomo --model_size 7b
```
