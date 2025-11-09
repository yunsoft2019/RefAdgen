from pathlib import Path
import torch
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


class BuildProduct:
    def __init__(self, image_file, image_type):
        self.image_file = image_file
        self.image_type = image_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.box = self.get_dino_box()
        self.ad_image = Image.open(self.image_file)
        self.masks = self.get_masks()

    def get_dino_box(self):
        dino_path = "/mnt/c/Projects/ModelDebugging/GroundingDINO/"
        dino_model = load_model(
            dino_path + "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            dino_path + "weights/groundingdino_swint_ogc.pth"
        )
        image_source, image = load_image(self.image_file)
        boxes, logits, phrases = predict(
            model=dino_model,
            image=image,
            caption=self.image_type,
            box_threshold=0.35,
            text_threshold=0.25
        )
        boxes = boxes.to(self.device)
        h, w, _ = image_source.shape
        scale_tensor = torch.Tensor([w, h, w, h]).to(self.device)
        boxes = boxes * scale_tensor
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
        box = boxes[0].tolist()
        return box

    def get_masks(self):
        image = np.array(self.ad_image.convert("RGB"))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            checkpoint = "/mnt/c/Projects/ModelDebugging/sam2/checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                multimask_output=False,
                box=self.box,
            )
            mask_tensor = torch.from_numpy(masks).unsqueeze(1).float()
            mask_tensor = mask_tensor.to(self.device)
            return masks, mask_tensor

    def get_product(self, masks):
        image_rgba = self.ad_image.convert("RGBA")
        image_array = np.array(image_rgba)
        mask_np = (masks[0] * 255).astype(np.uint8)
        result_array = np.ones_like(image_array, dtype=np.uint8) * 255
        for channel in range(3):
            result_array[:, :, channel] = np.where(
                mask_np > 0,
                image_array[:, :, channel],
                255
            )
        result_array[:, :, 3] = 255
        product_image = Image.fromarray(result_array)
        product_image = product_image.convert("RGB")
        return product_image

    def execute(self):
        masks, mask_tensor = self.get_masks()
        product_image = self.get_product(masks)

        return product_image, mask_tensor, product_image.size


if __name__ == '__main__':
    img_file = Path("../../data_sources/Ad Image/1737963914_7116103.png")
    print(img_file.exists())
    img_type = "backpack"
    build_product = BuildProduct(img_file, img_type)
    p_image, m_tensor, size = build_product.execute()
