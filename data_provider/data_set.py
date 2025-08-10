import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as tf
from PIL import Image
from transformers import CLIPImageProcessor

from random import choice


class GenDataset(Dataset):
    def __init__(
            self,
            json_file,
            tokenizer,
            size=512,
            image_root_path="",
    ):

        with open(json_file, "r") as f:
            self.data = json.load(f)
        # self.data = self.data[:99]
        self.tokenizer = tokenizer
        self.size = size
        self.image_root_path = image_root_path
        self.geometric_transform = transforms.Compose([
            transforms.Resize(640, interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        self.image_pixel_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.mask_pixel_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        item = self.data[idx]
        ad_image_path = self.image_root_path + "/Ad Image/" + item["original_image"]
        ad_image = Image.open(ad_image_path).convert("RGB")
        product_image_path = self.image_root_path + "/Product Image/" + item["mask_image"]
        product_image = Image.open(product_image_path).convert("RGB")
        mask_image_np = np.load(self.image_root_path + "/mask_npy/" + item["mask_npy"])
        mask_image = Image.fromarray((mask_image_np * 255).astype(np.uint8)).convert("L")
        ad_image = self.geometric_transform(ad_image)
        product_image = self.geometric_transform(product_image)
        mask_image = self.geometric_transform(mask_image)
        crop_params = transforms.RandomCrop.get_params(
            ad_image, output_size=(640, 512)
        )
        i, j, h, w = crop_params

        ad_image = tf.crop(ad_image, i, j, h, w)
        product_image = tf.crop(product_image, i, j, h, w)
        mask_image = tf.crop(mask_image, i, j, h, w)

        vae_ad_image = self.image_pixel_transform(ad_image)
        vae_product_image = self.image_pixel_transform(product_image)

        vae_mask_image = self.mask_pixel_transform(mask_image)
        vae_mask_image = (vae_mask_image > 0.5).float()

        text = choice(item['image_text'])

        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < 0.05:
            drop_image_embed = 1
        elif rand_num < 0.1:
            text = ""
        elif rand_num < 0.15:
            text = ""
            drop_image_embed = 1

        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        null_text_input_ids = self.tokenizer(
            "",
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        clip_image = self.clip_image_processor(images=product_image, return_tensors="pt").pixel_values

        return {
            "vae_ad_image": vae_ad_image,
            "vae_product_image": vae_product_image,
            "vae_mask_image": vae_mask_image,  # 新增返回掩码
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "text": text,
            "text_input_ids": text_input_ids,
            "null_text_input_ids": null_text_input_ids,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    vae_ad_images = torch.stack([example["vae_ad_image"] for example in data]).to(
        memory_format=torch.contiguous_format).float()
    vae_product_images = torch.stack([example["vae_product_image"] for example in data]).to(
        memory_format=torch.contiguous_format).float()
    vae_mask_image = torch.stack([example["vae_mask_image"] for example in data]).to(  # 堆叠掩码
        memory_format=torch.contiguous_format).float()

    clip_image = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embed = [example["drop_image_embed"] for example in data]

    text = [example["text"] for example in data]
    input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    null_input_ids = torch.cat([example["null_text_input_ids"] for example in data], dim=0)

    return {
        "vae_ad_images": vae_ad_images,
        "vae_product_images": vae_product_images,
        "vae_mask_images": vae_mask_image,
        "clip_image": clip_image,
        "drop_image_embed": drop_image_embed,
        "text": text,
        "input_ids": input_ids,
        "null_input_ids": null_input_ids,
    }
