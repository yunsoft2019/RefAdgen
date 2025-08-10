# 01 Function Introduction
This project is named RefAdGen, a deep learning project for generating advertising images. Its core function is to preserve the details of the original product image when generating new ad images.

The key components and workflow of the project are as follows:

Training: The project includes scripts and tools for model training. It uses a model called ADGenModel and utilizes stable-diffusion-v1-5 as the base model.

Data Processing: The GenDataset class is used to process the dataset. The dataset contains original ad images, product images, and mask images, as well as text descriptions. During training, these images undergo various transformations and cropping.

Model Architecture: The project uses multiple pre-trained models to work together:

VAE (Variational Autoencoder): Used for encoding and decoding images between pixel and latent spaces.

CLIPTextModel and CLIPVisionModelWithProjection: Serve as the text encoder and image encoder, respectively, to understand text prompts and extract image features.

UNet: A crucial part of the diffusion model.

IP-Adapter: Used to fuse image features with the main generative network to guide the generation process and maintain consistency in image content or style.

Prediction: The project also includes a Predictor class for inference and prediction. During the prediction phase, it uses tools like GroundingDINO and SAM2 (Segment Anything Model 2) to locate and segment products in the image, thereby allowing for the precise preservation of these products when generating new ad images.

# 02 Project Deployment

## 02.01 Project operating environment
    GPU: 5090 D
    PyTorch: torch2.7.1+cu128
## 02.02 installation package
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install transformers
    pip install accelerate
    pip install diffusers
    pip install sam2
    pip install qwen-vl-utils[decord]
    Download and install GroundingDINO from https://github.com/IDEA-Research/GroundingDINO
