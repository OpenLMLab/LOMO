set -x
port=$(shuf -i25000-30000 -n1)

# for full parameter fine-tuning using LOMO
deepspeed --master_port "$port" --include localhost:0 src/train_lomo.py config/args_lomo.yaml

# for LoRA + LOMO
#deepspeed --master_port "$port" --include localhost:0 src/train_lomo_lora.py config/args_lomo_lora.yaml
