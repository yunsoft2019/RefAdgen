from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent.absolute().as_posix()
sys.path.append(BASE_DIR)
# 添加 GroundingDINO 路径（尝试两种可能的目录名）
for dir_name in ["external_models", "exteral_models"]:
    GROUNDINGDINO_PATH = Path(BASE_DIR) / dir_name / "GroundingDINO"
    if GROUNDINGDINO_PATH.exists():
        sys.path.insert(0, str(GROUNDINGDINO_PATH))
        break
# 添加 SAM2 路径（尝试多种可能的目录名）
for dir_name in ["external_models", "exteral_models"]:
    SAM2_PATH = Path(BASE_DIR) / dir_name / "SAM2"
    if SAM2_PATH.exists():
        sys.path.insert(0, str(SAM2_PATH))
        break
# 如果 SAM2 在项目根目录
SAM2_ROOT = Path(BASE_DIR) / "sam2"
if SAM2_ROOT.exists():
    sys.path.insert(0, str(SAM2_ROOT))
# 尝试绝对路径（从 build_product.py 中看到的路径）
SAM2_ABS_PATH = Path("/mnt/c/Projects/ModelDebugging/sam2")
if SAM2_ABS_PATH.exists():
    sys.path.insert(0, str(SAM2_ABS_PATH))

from experiments.arguments import parse_args
from experiments.predict_utils import PredictUtils


class Predictor:
    def __init__(self, args):
        self.args = args
        predict_utils = PredictUtils(args)
        vae, tokenizer, text_encoder, image_encoder, unet = predict_utils.init_models()
        image_proj = predict_utils.get_resampler(unet, image_encoder)
        unet = predict_utils.set_unet_attn(unet)
        adapter_modules = predict_utils.get_adapter_modules(unet)
        ref_unet = predict_utils.get_ref_unet()
        unet, ref_unet, image_proj, adapter_modules = predict_utils.load_state(
            unet, ref_unet, image_proj, adapter_modules
        )
        noise_scheduler = predict_utils.get_noise_scheduler()
        self.pipe = predict_utils.get_gen_pipe(
            unet, ref_unet, vae, tokenizer, text_encoder, image_encoder, image_proj, noise_scheduler
        )
        self.gen_transform = predict_utils.get_transform()
        self.clip_image_processor = predict_utils.get_clip_image_processor()
        self.product_image, self.mask_tensor, self.size = predict_utils.build_product()
        self.generator = predict_utils.get_generator()
        self.prompt = args.prompt
        self.output_dir = args.output_dir
        self.model_ckpt = args.model_ckpt
        self.image_file = args.image_file
        self.num_samples = args.num_samples

    def execute(self):
        prompt = self.prompt
        prompt = prompt + ', best quality, high quality'
        null_prompt = ''
        negative_prompt = 'bare, naked, nude, undressed, monochrome, bad anatomy, worst quality, low quality'
        vae_product_images = self.gen_transform(self.product_image).unsqueeze(0)
        ref_clip_image = self.clip_image_processor(images=self.product_image, return_tensors="pt").pixel_values
        outputs = self.pipe(
            ref_image=vae_product_images,
            prompt=prompt,
            ref_clip_image=ref_clip_image,
            null_prompt=null_prompt,
            negative_prompt=negative_prompt,
            mask_image_tensor=self.mask_tensor,
            width=512,
            height=640,
            num_images_per_prompt=self.num_samples,
            guidance_scale=7.5,
            image_scale=1.0,
            generator=self.generator,
            num_inference_steps=30,
        ).images
        for idx, output in enumerate(outputs):
            output_image = output.resize(self.size)
            save_path = Path(self.output_dir + '/' + self.model_ckpt) / "images"
            save_path.mkdir(parents=True, exist_ok=True)
            save_file = save_path / (Path(self.image_file).name[:-4] + "_" + format(idx + 1, "03d") + ".png")
            output_image.save(save_file.as_posix())


if __name__ == '__main__':
    configs = parse_args()
    predictor = Predictor(configs)
    predictor.execute()
