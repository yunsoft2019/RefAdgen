import time
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent.absolute().as_posix()
sys.path.append(BASE_DIR)

import torch.nn.functional as fun
import torch

from experiments.arguments import parse_args
from experiments.train_utils import TrainUtils
from utils.base_utils import BaseUtils


class Trainer:
    def __init__(self, args):
        train_utils = TrainUtils(args)
        accelerator, self.logger = train_utils.get_accelerator()
        tokenizer, text_encoder, unet, vae, image_encoder = train_utils.init_models()
        unet = train_utils.add_unet_channel(unet)
        image_proj = train_utils.get_resampler(unet, image_encoder)
        unet = train_utils.set_unet_attn(unet)
        adapter_modules = train_utils.get_adapter_modules(unet)
        ref_unet = train_utils.get_ref_unet()
        vae, text_encoder, image_encoder, image_proj, ref_unet, adapter_modules, unet = train_utils.set_requires(
            vae, text_encoder, image_encoder, image_proj, ref_unet, adapter_modules, unet
        )
        ad_gen_model = train_utils.get_ad_gen_model(unet, ref_unet, image_proj, adapter_modules)
        optimizer = train_utils.get_optimizer(accelerator, ad_gen_model, unet)
        self.noise_scheduler = train_utils.get_noise_scheduler()
        train_dataloader = train_utils.get_train_dataloader(accelerator, tokenizer)
        lr_scheduler = train_utils.get_lr_scheduler(accelerator, optimizer)
        self.ad_gen_model, self.optimizer, self.train_dataloader, self.lr_scheduler = train_utils.accelerator_prepare(
            accelerator, ad_gen_model, optimizer, train_dataloader, lr_scheduler
        )
        self.weight_dtype = train_utils.get_weight_dtype(accelerator)
        self.text_encoder, self.vae, self.image_encoder = train_utils.set_dtype(
            accelerator, text_encoder, vae, image_encoder, self.weight_dtype
        )
        self.checkpointing_steps_val = train_utils.get_checkpointing_steps_val()
        self.accelerator = train_utils.init_trackers(accelerator)
        self.starting_epoch, self.global_steps, self.cost_time = train_utils.load_training_checkpoint(
            self.ad_gen_model,
            args.output_dir,
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True}
        )
        self.train_epochs = args.train_epochs
        self.noise_offset = args.noise_offset
        self.snr_gamma = args.snr_gamma
        self.batch_size = args.batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.output_dir = args.output_dir
        self.max_train_steps = args.max_train_steps
        self.train_utils = train_utils

    def execute(self):
        epoch_counter = 0
        start_time = time.perf_counter() if self.cost_time == 0 else time.perf_counter() - self.cost_time
        for epoch_idx in range(self.starting_epoch, self.train_epochs):
            epoch_counter = epoch_idx
            self.ad_gen_model.train()
            train_loss = 0.0
            if hasattr(self.train_dataloader.sampler, "set_epoch") and isinstance(self.train_dataloader.sampler,
                                                                                  torch.utils.data.distributed.DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch_idx)

            for step, batch in enumerate(self.train_dataloader):
                current_iter_begin_time = time.perf_counter()
                with self.accelerator.accumulate(self.ad_gen_model):

                    with torch.no_grad():
                        latents = self.vae.encode(
                            batch["vae_ad_images"].to(self.accelerator.device,
                                                      dtype=self.weight_dtype)).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor

                        target_latent_h = latents.shape[2]
                        target_latent_w = latents.shape[3]

                        ref_latents = self.vae.encode(
                            batch["vae_product_images"].to(self.accelerator.device,
                                                           dtype=self.weight_dtype)).latent_dist.sample()
                        ref_latents = ref_latents * self.vae.config.scaling_factor

                        vae_mask = batch["vae_mask_images"].to(self.accelerator.device, dtype=self.weight_dtype)
                        if vae_mask.ndim == 3:
                            vae_mask = vae_mask.unsqueeze(1)
                        mask_latent = fun.interpolate(vae_mask, size=(target_latent_h, target_latent_w), mode='nearest')

                    noise = torch.randn_like(latents)
                    if self.noise_offset > 0:
                        noise += self.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1),
                            device=latents.device, dtype=latents.dtype
                        )
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    unet_input = torch.cat([noisy_latents, mask_latent], dim=1)

                    clip_images = []
                    for clip_image_item, drop_image_embed_item in zip(batch["clip_image"], batch["drop_image_embed"]):
                        if drop_image_embed_item == 1:
                            clip_images.append(torch.zeros_like(clip_image_item))
                        else:
                            clip_images.append(clip_image_item)
                    clip_images_tensor_batch = torch.stack(clip_images, dim=0)

                    with torch.no_grad():
                        image_encoder_output = self.image_encoder(
                            clip_images_tensor_batch.to(self.accelerator.device, dtype=self.weight_dtype),
                            output_hidden_states=True)
                        if hasattr(image_encoder_output,
                                   "hidden_states") and image_encoder_output.hidden_states is not None and len(
                            image_encoder_output.hidden_states) >= 2:
                            image_embeds = image_encoder_output.hidden_states[-2]
                        else:
                            image_embeds = image_encoder_output.last_hidden_state

                    with torch.no_grad():
                        encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.accelerator.device))[0]

                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    prepared_ref_item_mask_train = None

                    model_pre = self.ad_gen_model(
                        encoder_hidden_states,
                        unet_input,
                        ref_latents,
                        image_embeds,
                        timesteps,
                        reference_item_attention_mask=prepared_ref_item_mask_train
                    )
                    if self.snr_gamma == 0:
                        loss = fun.mse_loss(
                            model_pre.float(), target.float(), reduction="mean"
                        )
                    else:
                        snr = self.train_utils.compute_snr(self.noise_scheduler, timesteps)
                        if self.noise_scheduler.config.prediction_type == "v_prediction":
                            snr = snr + 1
                        mse_loss_weights = (
                                torch.stack(
                                    [snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1
                                ).min(dim=1)[0]
                                / snr
                        )
                        loss = fun.mse_loss(
                            model_pre.float(), target.float(), reduction="none"
                        )
                        loss = (
                                loss.mean(dim=list(range(1, len(loss.shape))))
                                * mse_loss_weights
                        )
                        loss = loss.mean()

                    avg_loss = self.accelerator.gather(loss.repeat(self.batch_size)).mean()
                    train_loss += avg_loss.item()

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        self.accelerator.log({
                            "train_loss": train_loss / self.gradient_accumulation_steps if self.gradient_accumulation_steps > 0 else train_loss},
                            step=self.global_steps)
                        if self.accelerator.is_main_process:
                            spend_time = time.perf_counter() - start_time
                            self.logger.info(
                                "Epoch {:03d}, step {:08d},  step_loss: {:.12f}, lr: {:.12f}, time: {:.8f}, spend_time: {}, speed: {:.6f}".format(
                                    epoch_counter, self.global_steps, loss.detach().item(),
                                    self.lr_scheduler.get_last_lr()[0],
                                    time.perf_counter() - current_iter_begin_time, BaseUtils.seconds_to_hms(spend_time),
                                    spend_time / (self.global_steps + 1))
                            )
                        train_loss = 0.0

                    self.global_steps += 1
                    if isinstance(self.checkpointing_steps_val, int):
                        if self.global_steps % self.checkpointing_steps_val == 0:
                            cost_time = time.perf_counter() - start_time
                            self.train_utils.checkpoint_model(self.output_dir, self.global_steps, self.ad_gen_model,
                                                              epoch_counter, self.global_steps, cost_time)

                    if self.global_steps >= self.max_train_steps:
                        break
                if self.global_steps >= self.max_train_steps:
                    break
            if self.accelerator.is_main_process and self.checkpointing_steps_val == "epoch":
                cost_time = time.perf_counter() - start_time
                self.train_utils.checkpoint_model(self.output_dir, self.global_steps, self.ad_gen_model, epoch_counter,
                                                  self.global_steps, cost_time)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            cost_time = time.perf_counter() - start_time
            self.train_utils.checkpoint_model(self.output_dir, self.global_steps, self.ad_gen_model, epoch_counter,
                                              self.global_steps, cost_time)
        self.accelerator.end_training()


if __name__ == '__main__':
    params = parse_args()
    trainer = Trainer(params)
    trainer.execute()
