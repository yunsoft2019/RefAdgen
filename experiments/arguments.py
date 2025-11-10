import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="RefAdGen: A training script for generating advertisement images with preserved product details.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="This parameter refers to 'stable-diffusion-v1-5' during training and 'Realistic_Vision_V4.0_noVAE' during inference/testing.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        
        help="Specifies the path to the pretrained image encoder model, which is used for extracting image features.",
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default=None,
        help="Specifies the path to the pretrained VAE model, which is responsible for encoding and decoding images between pixel space and latent space.",
    )

    parser.add_argument(
        "--adapter_model_path",
        type=str,
        default=None,
        help="Specifies the path to the pretrained IP-Adapter model, which is used to fuse image features with the main generation network to guide the generation process and maintain consistency in image content or style.",
    )

    parser.add_argument(
        "--dataset_json_path",
        type=str,
        default=None,
        help="Specifies the path to the JSON file containing dataset information.",
    )

    parser.add_argument(
        "--train_data_path",
        type=str,
        default=None,
        help="Specifies the root directory of the dataset.",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        '--clip_penultimate',
        type=bool,
        default=False,
        help='Specifies whether to use the hidden states from the penultimate layer of the CLIP image encoder as image embeddings.'
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Sets the batch size per device during training loader."
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0.05,
        help="By adding a small offset to the noise in the diffusion process, the quality of generated images is improved, especially when dealing with extremely bright or dark areas."
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=0, help="Adjusts the loss weights at different noise levels to improve the quality of generated images, especially in enhancing image details and clarity."
    )

    parser.add_argument("--train_epochs", type=int, default=100000,help="Sets the total number of full training epochs the model will undergo.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Sets the maximum number of update steps to be performed during model training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=64,
        help="Sets the number of gradient accumulation steps before performing a parameter update, which helps simulate larger training batch sizes with limited GPU memory.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Sets the initial learning rate for model training, which determines the magnitude of parameter adjustments during each update.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Sets the strength of L2 regularization (weight decay), which helps prevent model overfitting by penalizing large weight values.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        help="Sets the type of learning rate scheduler used to dynamically adjust the learning rate during training.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--num_warmup_steps", type=int, default=500, help="Sets the number of steps for the learning rate warm-up phase, during which the learning rate gradually increases from a smaller value to the initial learning rate."
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Sets the directory for saving logs and training progress.",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Specifies the target platform for reporting metrics and logs during training.[tensorboard, wandb, all, none]",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Sets the frequency, in training steps, for saving model checkpoints.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="Identifies the local rank of the current process on the current node, typically set automatically by the launch script.")
    parser.add_argument(
        '--model_ckpt',
        default="Specifies the path to the pretrained model checkpoint for loading the model's initial weights for training.",
        
        type=str
    )
    parser.add_argument('--image_file', type=str, help="Input image file path and name.")
    parser.add_argument('--image_type', type=str, help="Specifies the image type.")
    parser.add_argument('--output_dir', type=str, default="./outputs",help="Output file root directory.")
    parser.add_argument('--device', type=str, default="cuda:0",help="Specifies the device.")
    parser.add_argument('--prompt', type=str,  default="Input image file description.")
    parser.add_argument('--num_samples', type=int,  default=1000,help="Number of images generated per inference.")
    parser.add_argument(
        "--sam2_repo_id",
        type=str,
        default="facebook/sam2.1-hiera-large",
        help="Hugging Face repo_id for SAM2 checkpoint (used to download sam2.1_hiera_large.pt)."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
