set -x
port=$(shuf -i25000-30000 -n1)

# for full parameter fine-tuning using LOMO
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 src/train_zero.py config/hf_args_zero.yaml

# for LoRA + LOMO
#deepspeed --master_port "$port" --include localhost:0 src/train_zero_lora.py config/hf_args_zero_lora.yaml
