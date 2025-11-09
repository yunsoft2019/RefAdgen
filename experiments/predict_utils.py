from pathlib import Path
from PIL import Image
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from utils.resampler import Resampler
from utils.attention_processor import HiddenStateCacheAttnProcessor, ScaledDotProductAttentionProcessor, \
    AttentionFusionModel
from utils.gen_pipe import GenPipe
from data_provider.build_product import BuildProduct


class PredictUtils:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        # 检查是否为 HuggingFace 模型 ID（包含 / 但不以 / 开头，且路径不存在）
        def is_hf_model_id(path):
            return "/" in path and not path.startswith("/") and not Path(path).exists()
        
        # 处理 base_model_path
        if is_hf_model_id(args.base_model_path):
            self.base_model_path = args.base_model_path  # HuggingFace 模型 ID
        else:
            self.base_model_path = str(Path(args.base_model_path).resolve())  # 本地路径
        
        # 处理 vae_model_path
        if is_hf_model_id(args.vae_model_path):
            self.vae_model_path = args.vae_model_path  # HuggingFace 模型 ID
        else:
            self.vae_model_path = str(Path(args.vae_model_path).resolve())  # 本地路径
        
        # 处理 image_encoder_path：支持 HuggingFace 模型 ID 或子目录路径
        if is_hf_model_id(args.image_encoder_path):
            self.image_encoder_path = args.image_encoder_path  # HuggingFace 模型 ID 或子目录路径
        else:
            self.image_encoder_path = str(Path(args.image_encoder_path).resolve())  # 本地路径
        
        self.output_dir = args.output_dir
        self.model_ckpt = args.model_ckpt
        self.image_file = args.image_file
        self.image_type = args.image_type

    @staticmethod
    def resize_img(input_image, max_side=640, min_side=512, mode=Image.Resampling.BILINEAR, base_pixel_number=64):
        w, h = input_image.size
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
        input_image = input_image.resize([w_resize_new, h_resize_new], mode)

        return input_image

    def init_models(self):
        vae = AutoencoderKL.from_pretrained(self.vae_model_path).to(dtype=torch.float16,
                                                                               device=self.device)
        tokenizer = CLIPTokenizer.from_pretrained(self.base_model_path,
                                                  subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.base_model_path,
                                                     subfolder="text_encoder").to(
            dtype=torch.float16, device=self.device)
        
        # 处理 image_encoder_path：如果是子目录格式，解析为 repo_id 和 subfolder
        image_encoder_path = self.image_encoder_path
        image_encoder_subfolder = None
        if "/" in self.image_encoder_path and not self.image_encoder_path.startswith("/") and not Path(self.image_encoder_path).exists():
            parts = self.image_encoder_path.split("/")
            if len(parts) >= 3:
                # 格式：org/repo/subfolder/path
                repo_id = f"{parts[0]}/{parts[1]}"
                subfolder = "/".join(parts[2:])
                image_encoder_path = repo_id
                image_encoder_subfolder = subfolder
        
        if image_encoder_subfolder:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, subfolder=image_encoder_subfolder).to(
            dtype=torch.float16, device=self.device)
        else:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(
            dtype=torch.float16, device=self.device)
        unet_model_path = self.base_model_path
        unet_subfolder = "unet"
        unet_config = UNet2DConditionModel.load_config(unet_model_path, subfolder=unet_subfolder)
        unet_config["in_channels"] = 5
        unet = UNet2DConditionModel.from_config(unet_config).to(dtype=torch.float16, device=self.device)
        return vae, tokenizer, text_encoder, image_encoder, unet

    def get_resampler(self, unet, image_encoder):
        image_proj = Resampler(
            dim=unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=16,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4
        )
        image_proj = image_proj.to(dtype=torch.float16, device=self.device)
        return image_proj

    @staticmethod
    def set_unet_attn(unet):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                hidden_size = 0
            if cross_attention_dim is None:
                attn_procs[name] = AttentionFusionModel(name, hidden_size)
            else:
                attn_procs[name] = ScaledDotProductAttentionProcessor(name, hidden_size=hidden_size,
                                                                      cross_attention_dim=cross_attention_dim)

        unet.set_attn_processor(attn_procs)
        return unet

    def get_adapter_modules(self, unet):
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        adapter_modules = adapter_modules.to(dtype=torch.float16, device=self.device)
        return adapter_modules

    def get_ref_unet(self):
        ref_unet = UNet2DConditionModel.from_pretrained(
            self.base_model_path,
            subfolder="unet"
        ).to(
            dtype=torch.float16,
            device=self.device
        )
        ref_unet.set_attn_processor(
            {name: HiddenStateCacheAttnProcessor() for name in ref_unet.attn_processors.keys()}
        )
        return ref_unet

    def load_state(self, unet, ref_unet, image_proj, adapter_modules):
        # DeepSpeed 保存格式: outputs/checkpoint-{step}/pytorch_model/mp_rank_00_model_states.pt
        model_ckpt = self.output_dir + '/checkpoint-' + self.model_ckpt + '/pytorch_model/mp_rank_00_model_states.pt'
        model_sd = torch.load(model_ckpt, map_location="cpu")["module"]

        ref_unet_dict = {}
        unet_dict = {}
        image_proj_dict = {}
        adapter_modules_dict = {}
        for k in model_sd.keys():
            if k.startswith("ref_unet"):
                ref_unet_dict[k.replace("ref_unet.", "")] = model_sd[k]
            elif k.startswith("unet"):
                unet_dict[k.replace("unet.", "")] = model_sd[k]
            elif k.startswith("proj"):
                image_proj_dict[k.replace("proj.", "")] = model_sd[k]
            elif k.startswith("adapter_modules"):
                adapter_modules_dict[k.replace("adapter_modules.", "")] = model_sd[k]
            else:
                print(k)

        if 'conv_in.weight' in unet_dict and unet_dict['conv_in.weight'].shape[1] == 4:
            original_conv_in_weight = unet_dict['conv_in.weight']
            new_conv_in_weight = torch.zeros(
                original_conv_in_weight.shape[0],
                5,
                original_conv_in_weight.shape[2],
                original_conv_in_weight.shape[3],
                dtype=original_conv_in_weight.dtype,
                device=original_conv_in_weight.device
            )
            new_conv_in_weight[:, :4, :, :] = original_conv_in_weight

            unet_dict['conv_in.weight'] = new_conv_in_weight

        ref_unet.load_state_dict(ref_unet_dict)
        image_proj.load_state_dict(image_proj_dict)
        adapter_modules.load_state_dict(adapter_modules_dict)
        unet.load_state_dict(unet_dict)
        return unet, ref_unet, image_proj, adapter_modules

    @staticmethod
    def get_noise_scheduler():
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        return noise_scheduler

    @staticmethod
    def get_gen_pipe(unet, ref_unet, vae, tokenizer, text_encoder, image_encoder, image_proj, noise_scheduler):
        pipe = GenPipe(unet=unet, reference_unet=ref_unet, vae=vae, tokenizer=tokenizer,
                       text_encoder=text_encoder, image_encoder=image_encoder,
                       ImgProj=image_proj,
                       scheduler=noise_scheduler,
                       safety_checker=StableDiffusionSafetyChecker,
                       feature_extractor=CLIPImageProcessor)
        return pipe

    @staticmethod
    def get_transform():
        gen_transform = transforms.Compose([
            transforms.Resize([640, 512], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return gen_transform

    @staticmethod
    def get_clip_image_processor():
        clip_image_processor = CLIPImageProcessor()
        return clip_image_processor

    def build_product(self):
        build_product = BuildProduct(self.image_file, self.image_type)
        product_image, mask_tensor, size = build_product.execute()
        product_image = self.resize_img(product_image)
        mask_tensor = mask_tensor.to(self.device)
        return product_image, mask_tensor, size

    def get_generator(self):
        generator = torch.Generator(device=self.device).manual_seed(42)
        return generator
