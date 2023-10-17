import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

'''
This is the code for merging the LoRA adapter with the base model. [ref] https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py

To load `lora + lomo` checkpoint, please first run `python merge_llama_with_lora.py` to merge the weights. Then, set `resume_from_checkpoint` to the merged weights path.
'''


def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, low_cpu_mem_usage=True
    )
    # base_tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    # base_tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name_or_path", type=str, required=True)
    # parser.add_argument("--output_path", type=str, required=True)
    # parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--llama", action="store_true", required=True)

    # model_name_or_path = '/remote-home/share/llama_hf/7B'
    model_name_or_path = 'outputs/wic_7B_lora-qv-r2-lomo/output_lr0.005_bs16_warmup0.05_clipnorm1.0/checkpoint-0'
    # good path
    ckpt_path = 'outputs/wic_7B_lora-qv-r2-lomo/output_lr0.005_bs16_warmup0.05_clipnorm1.0/checkpoint-0'
    # ckpt_path = 'outputs/lora65b_checkpoint-2500'
    lora_path = os.path.join(ckpt_path, 'adapter_model')
    output_path = os.path.join(ckpt_path, 'merge_weights')

    # lora_path = 'outputs/belle_llama-7b_1w_zh_len1400_zero2_5e4/output_lora_adamw_hf_lr0.0005/checkpoint-1560/global_step1560/adapter_model'
    # output_path = 'outputs/belle_llama-7b_1w_zh_len1400_zero2_5e4/output_lora_adamw_hf_lr0.0005/checkpoint-1560/global_step1560/merge_weights'

    apply_lora(model_name_or_path, output_path, lora_path)
    