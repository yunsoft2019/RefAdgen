python experiments/predict.py \
  --base_model_path="SG161222/Realistic_Vision_V5.1_noVAE" \
  --image_encoder_path="h94/IP-Adapter/models/image_encoder" \
  --vae_model_path="stabilityai/sd-vae-ft-mse" \
  --image_file="../../datasets/Ad Image/01.png" \
  --image_type="backpack" \
  --prompt="A backpack rests on the green lawn in a park, surrounded by lush grass, red flowers, and green leaves." \
  --model_ckpt="900000" \
  --num_samples=3