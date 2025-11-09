export HOST_NUM=1
accelerate launch \
  --gpu_ids 0 \
  --use_deepspeed \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision 'fp16' \
  --dynamo_backend 'no' \
  --deepspeed_config_file "./jsons/zero_stage2_config.json" \
  experiments/train.py \
  --base_model_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --vae_model_path="stabilityai/sd-vae-ft-mse" \
  --adapter_model_path="h94/IP-Adapter/models/ip-adapter-plus_sd15.bin" \
  --image_encoder_path="h94/IP-Adapter/models/image_encoder" \
  --dataset_json_path="./jsons/train_sd.json" \
  --train_data_path="./data_sources" \
  --clip_penultimate=False \
  --batch_size=3 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=100000000 \
  --learning_rate=1e-5 \
  --weight_decay=0.01 \
  --lr_scheduler="constant" \
  --num_warmup_steps=2000 \
  --output_dir="outputs" \
  --checkpointing_steps=100