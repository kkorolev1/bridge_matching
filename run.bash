export WANDB_API_KEY="24fb79a2b575b1f5b45d3e9f6955597fc65b9477" 
PYTHONPATH=".:./edm" 
accelerate launch --config_file accelerate_config.yaml --num_processes 1 train.py --config-name ffhq_inpainting.yaml 