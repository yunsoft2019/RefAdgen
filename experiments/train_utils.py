from pathlib import Path
import logging
import itertools

import torch
import transformers
import datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyOptim, DummyScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler

from utils.resampler import Resampler
from utils.attention_processor import HiddenStateCacheAttnProcessor, ScaledDotProductAttentionProcessor, \
    AttentionFusionModel
from models.ad_gen_model import ADGenModel
from data_provider.data_loader import get_loader

logger = get_logger(__name__)


class TrainUtils:
    def __init__(self, args):
        self.args = args
        self.logging_dir = Path(args.output_dir) / args.logging_dir
        self.report_to = args.report_to
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        # 确保路径是绝对路径，并去掉末尾的斜杠
        self.base_model_path = str(Path(args.base_model_path).resolve())
        self.vae_model_path = str(Path(args.vae_model_path).resolve())
        self.image_encoder_path = str(Path(args.image_encoder_path).resolve())
        self.adapter_model_path = args.adapter_model_path
        self.lr_scheduler = args.lr_scheduler
        self.num_warmup_steps = args.num_warmup_steps
        self.max_train_steps = args.max_train_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.checkpointing_steps = args.checkpointing_steps
        if args.seed is not None:
            set_seed(args.seed)

    @staticmethod
    def count_model_params(model):
        return sum([p.numel() for p in model.parameters()]) / 1e6

    @staticmethod
    def compute_snr(noise_scheduler, timesteps):
        alphas_cum_prod = noise_scheduler.alphas_cum_prod
        sqrt_alphas_cum_prod = alphas_cum_prod ** 0.5
        sqrt_one_minus_alphas_cum_prod = (1.0 - alphas_cum_prod) ** 0.5
        sqrt_alphas_cum_prod = sqrt_alphas_cum_prod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_alphas_cum_prod.shape) < len(timesteps.shape):
            sqrt_alphas_cum_prod = sqrt_alphas_cum_prod[..., None]
        alpha = sqrt_alphas_cum_prod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cum_prod = sqrt_one_minus_alphas_cum_prod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cum_prod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cum_prod = sqrt_one_minus_alphas_cum_prod[..., None]
        sigma = sqrt_one_minus_alphas_cum_prod.expand(timesteps.shape)
        snr = (alpha / sigma) ** 2
        return snr

    def get_accelerator(self):
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        accelerator = Accelerator(
            log_with=self.report_to,
            project_dir=self.logging_dir.as_posix(),
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )

        logging.basicConfig(
            format="",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
        return accelerator, logger

    def init_models(self):
        tokenizer = CLIPTokenizer.from_pretrained(self.base_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(self.base_model_path, subfolder="text_encoder")
        unet = UNet2DConditionModel.from_pretrained(self.base_model_path, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(self.vae_model_path)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path)
        return tokenizer, text_encoder, unet, vae, image_encoder

    @staticmethod
    def add_unet_channel(unet):
        original_conv_in = unet.conv_in
        if unet.config.in_channels == 4:
            new_conv_in = torch.nn.Conv2d(
                in_channels=5,
                out_channels=original_conv_in.out_channels,
                kernel_size=original_conv_in.kernel_size,
                stride=original_conv_in.stride,
                padding=original_conv_in.padding,
                bias=(original_conv_in.bias is not None)
            )
            with torch.no_grad():
                new_conv_in.weight.data[:, :4, :, :] = original_conv_in.weight.data.clone()
                new_conv_in.weight.data[:, 4:5, :, :] = torch.zeros_like(new_conv_in.weight.data[:, 4:5, :, :])
                if original_conv_in.bias is not None:
                    new_conv_in.bias.data = original_conv_in.bias.data.clone()
            unet.conv_in = new_conv_in
            unet.config.in_channels = 5
        return unet

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
        ipa_weight = torch.load(self.adapter_model_path, map_location="cpu")
        image_proj.load_state_dict(ipa_weight['image_proj'])
        return image_proj

    @staticmethod
    def set_unet_attn(unet):
        attn_procs = {}
        st = unet.state_dict()
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
                continue

            if cross_attention_dim is None:
                attn_procs[name] = AttentionFusionModel(name, hidden_size)
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ref.weight": st[layer_name + ".to_k.weight"],
                    "to_v_ref.weight": st[layer_name + ".to_v.weight"],
                }
                if attn_procs[name].to_k_ref.bias is not None and (layer_name + ".to_k.bias") in st:
                    weights["to_k_ref.bias"] = st[layer_name + ".to_k.bias"]
                if attn_procs[name].to_v_ref.bias is not None and (layer_name + ".to_v.bias") in st:
                    weights["to_v_ref.bias"] = st[layer_name + ".to_v.bias"]
                attn_procs[name].load_state_dict(weights, strict=False)
            else:
                attn_procs[name] = ScaledDotProductAttentionProcessor(name, hidden_size=hidden_size,
                                                                      cross_attention_dim=cross_attention_dim)
        unet.set_attn_processor(attn_procs)
        del st
        return unet

    @staticmethod
    def get_adapter_modules(unet):
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        return adapter_modules

    def get_ref_unet(self):
        ref_unet = UNet2DConditionModel.from_pretrained(self.base_model_path, subfolder="unet")
        ref_unet.set_attn_processor(
            {name: HiddenStateCacheAttnProcessor() for name in ref_unet.attn_processors.keys()})
        return ref_unet

    @staticmethod
    def set_requires(vae, text_encoder, image_encoder, image_proj, ref_unet, adapter_modules, unet):
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        image_encoder.requires_grad_(False)

        image_proj.requires_grad_(True)
        ref_unet.requires_grad_(True)
        adapter_modules.requires_grad_(True)
        unet.requires_grad_(False)
        for param in unet.conv_in.parameters():
            param.requires_grad = True
        for proc in unet.attn_processors.values():
            for param in proc.parameters():
                param.requires_grad = True
        return vae, text_encoder, image_encoder, image_proj, ref_unet, adapter_modules, unet

    @staticmethod
    def get_ad_gen_model(unet, ref_unet, image_proj, adapter_modules):
        ad_gen_model = ADGenModel(unet, ref_unet, image_proj, adapter_modules)
        return ad_gen_model

    def get_optimizer(self, accelerator, ad_gen_model, unet):
        params_to_opt = itertools.chain(ad_gen_model.proj.parameters(), ad_gen_model.ref_unet.parameters(),
                                        ad_gen_model.adapter_modules.parameters(), unet.conv_in.parameters())

        print(self.count_model_params(ad_gen_model.proj))
        print(self.count_model_params(ad_gen_model.ref_unet))
        print(self.count_model_params(ad_gen_model.adapter_modules))
        print(self.count_model_params(unet.conv_in))
        accelerator.print(
            "Trainable parameters: proj:{:.2f}M, ref_unet:{:.2f}M, adapter_modules:{:.2f}M, conv_in:{:.2f}M".format(
                self.count_model_params(ad_gen_model.proj), self.count_model_params(ad_gen_model.ref_unet),
                self.count_model_params(ad_gen_model.adapter_modules), self.count_model_params(unet.conv_in)))

        if (
                accelerator.state.deepspeed_plugin is None
                or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            optimizer = torch.optim.AdamW(params_to_opt, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = DummyOptim(
                params_to_opt,
                lr=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["lr"],
                weight_decay=accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["weight_decay"]
            )
        return optimizer

    @staticmethod
    def get_noise_scheduler():
        noise_scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing", prediction_type="epsilon",
        )
        return noise_scheduler

    def get_train_dataloader(self, accelerator, tokenizer):
        train_dataloader = get_loader(accelerator, tokenizer, self.args)
        return train_dataloader

    def get_lr_scheduler(self, accelerator, optimizer):
        if accelerator.state.deepspeed_plugin is not None:
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "gradient_accumulation_steps"] = self.gradient_accumulation_steps

        if (
                accelerator.state.deepspeed_plugin is None
                or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            lr_scheduler = get_scheduler(
                name=self.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.max_train_steps,
            )
        else:
            lr_scheduler = DummyScheduler(
                optimizer,
                warmup_num_steps=accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"]["params"][
                    "warmup_num_steps"]
            )

        if (
                accelerator.state.deepspeed_plugin is not None
                and accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto"
        ):
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"] = self.batch_size
        return lr_scheduler

    @staticmethod
    def accelerator_prepare(accelerator, ad_gen_model, optimizer, train_dataloader, lr_scheduler):
        ad_gen_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ad_gen_model, optimizer, train_dataloader, lr_scheduler
        )
        return ad_gen_model, optimizer, train_dataloader, lr_scheduler

    @staticmethod
    def get_weight_dtype(accelerator):
        weight_dtype = torch.float32
        if accelerator.state.deepspeed_plugin is None:
            if accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16
        else:
            if accelerator.state.deepspeed_plugin.deepspeed_config.get("fp16", {}).get("enabled", False):
                weight_dtype = torch.float16
            elif accelerator.state.deepspeed_plugin.deepspeed_config.get("bf16", {}).get("enabled", False):
                weight_dtype = torch.bfloat16
        return weight_dtype

    @staticmethod
    def set_dtype(accelerator, text_encoder, vae, image_encoder, weight_dtype):
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        image_encoder.to(accelerator.device, dtype=weight_dtype)
        return text_encoder, vae, image_encoder

    def get_checkpointing_steps_val(self):
        if hasattr(self.checkpointing_steps, "isdigit"):
            checkpointing_steps_val = self.checkpointing_steps
            if self.checkpointing_steps.isdigit():
                checkpointing_steps_val = int(self.checkpointing_steps)
        else:
            checkpointing_steps_val = None
        return checkpointing_steps_val

    def init_trackers(self, accelerator):
        if accelerator.is_main_process:
            accelerator.init_trackers("text2image", config=vars(self.args))
        return accelerator

    @staticmethod
    def load_training_checkpoint(model, load_dir, tag=None, **kwargs):
        if not Path(load_dir + "/latest").exists():
            return 0, 0, 0
        _, checkpoint_state_dict = model.load_checkpoint(load_dir, tag=tag, **kwargs)
        epoch = checkpoint_state_dict["epoch"]
        last_global_step = checkpoint_state_dict["last_global_step"]
        cost_time = checkpoint_state_dict["cost_time"]
        del checkpoint_state_dict
        return epoch, last_global_step,cost_time

    @staticmethod
    def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step,cost_time, **kwargs):
        checkpoint_state_dict = {
            "epoch": epoch,
            "last_global_step": last_global_step,
            "cost_time": cost_time,
        }
        checkpoint_state_dict.update(kwargs)

        success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
        status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
        if success:
            logging.info(f"Success {status_msg}")
        else:
            logging.warning(f"Failure {status_msg}")
        return
