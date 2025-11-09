import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.utils import logging
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline, \
    StableDiffusionPipelineOutput
from utils.attention_processor import HiddenStateCacheAttnProcessor, AttentionFusionModel
from utils.resampler import Resampler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

logger = logging.get_logger(__name__)


class GenPipe(StableDiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    
    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~StableDiffusionPipeline.enable_sequential_cpu_offload`], the execution device can only be inferred from
        Accelerate's module hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.unet.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.unet.device

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            reference_unet: UNet2DConditionModel,
            image_encoder: CLIPVisionModelWithProjection,
            ImgProj: Resampler,
            scheduler: DDIMScheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker: bool = False,
    ):
        _safety_checker_instance = safety_checker if not inspect.isclass(safety_checker) else None
        _feature_extractor_instance = feature_extractor if not inspect.isclass(feature_extractor) else None

        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=_safety_checker_instance,
            feature_extractor=_feature_extractor_instance,
            requires_safety_checker=requires_safety_checker
        )
        self.register_modules(
            reference_unet=reference_unet,
            image_encoder=image_encoder,
            ImgProj=ImgProj,
        )
        self._clip_skip = None
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            lora_scale: Optional[float] = None,
            clip_skip: Optional[int] = None,
    ):
        if lora_scale is not None and hasattr(self, "load_lora_weights"):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] > text_input_ids.shape[-1] and text_input_ids.shape[
                -1] == self.tokenizer.model_max_length:
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1: -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds_out = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds_out[0]
            else:
                prompt_embeds_out = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask,
                                                      output_hidden_states=True)
                pooled_prompt_embeds = prompt_embeds_out[0]
                if hasattr(prompt_embeds_out, "hidden_states") and prompt_embeds_out.hidden_states is not None:
                    prompt_embeds = prompt_embeds_out.hidden_states[-(clip_skip + 1)]
                    prompt_embeds = self.text_encoder.text_model.final_layer_norm(
                        prompt_embeds)  # Apply final layer norm
                else:
                    prompt_embeds = pooled_prompt_embeds

        prompt_embeds_dtype = self.text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            un_cond_tokens: List[str]
            if negative_prompt is None:
                un_cond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                un_cond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                un_cond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            un_cond_input = self.tokenizer(
                un_cond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                un_cond_attention_mask = un_cond_input.attention_mask.to(device)
            else:
                un_cond_attention_mask = None

            negative_prompt_embeds_out = self.text_encoder(
                un_cond_input.input_ids.to(device),
                attention_mask=un_cond_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds_out[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def _get_ip_adapter_image_embeds(self, ref_clip_image_input, device, dtype, num_images_per_prompt_eff,
                                     do_classifier_free_guidance):
        if ref_clip_image_input is None:
            return None, None
        if not isinstance(ref_clip_image_input, torch.Tensor):
            raise ValueError("ref_clip_image_input must be a preprocessed tensor.")
        image_encoder_input = ref_clip_image_input.to(device=device, dtype=dtype)
        if image_encoder_input.shape[0] != num_images_per_prompt_eff:
            if image_encoder_input.shape[0] == 1:
                image_encoder_input = image_encoder_input.repeat(num_images_per_prompt_eff, 1, 1, 1)
            else:
                raise ValueError(
                    f"Batch size of ref_clip_image_input ({image_encoder_input.shape[0]}) "
                    f"does not match effective batch size for prompts ({num_images_per_prompt_eff})."
                )

        image_embeds_cond_out = self.image_encoder(image_encoder_input, output_hidden_states=True)
        if hasattr(image_embeds_cond_out, "hidden_states") and image_embeds_cond_out.hidden_states is not None:
            image_embeds_cond = image_embeds_cond_out.hidden_states[-2]
        else:
            image_embeds_cond = image_embeds_cond_out.last_hidden_state

        product_proj_embed_cond = self.ImgProj(image_embeds_cond)

        if do_classifier_free_guidance:
            zeros_input = torch.zeros_like(image_encoder_input)
            image_embeds_un_cond_out = self.image_encoder(zeros_input, output_hidden_states=True)
            if hasattr(image_embeds_un_cond_out, "hidden_states") and image_embeds_un_cond_out.hidden_states is not None:
                image_embeds_un_cond = image_embeds_un_cond_out.hidden_states[-2]
            else:
                image_embeds_un_cond = image_embeds_un_cond_out.last_hidden_state

            product_proj_embed_un_cond = self.ImgProj(image_embeds_un_cond)
            return product_proj_embed_cond, product_proj_embed_un_cond
        else:
            return product_proj_embed_cond, None

    def set_scale(self, scale_value: float):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, AttentionFusionModel):
                attn_processor.scale = scale_value

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            ref_image: torch.FloatTensor,
            mask_image_tensor: Optional[torch.FloatTensor] = None,
            ref_clip_image: Optional[torch.FloatTensor] = None,
            reference_item_attention_mask: Optional[torch.FloatTensor] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            null_prompt: Optional[Union[str, List[str]]] = "",
            num_images_per_prompt: int = 1,
            image_scale: float = 1.0,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: Optional[int] = None,
    ):
        self.set_scale(image_scale)

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        device = self._execution_device

        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        effective_batch_size = batch_size * num_images_per_prompt
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance,
            negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip if clip_skip is not None else self._clip_skip
        )

        product_proj_embed_cond, product_proj_embed_un_cond = self._get_ip_adapter_image_embeds(
            ref_clip_image, device, prompt_embeds.dtype, effective_batch_size, do_classifier_free_guidance
        )

        if do_classifier_free_guidance:
            if product_proj_embed_cond is not None and product_proj_embed_un_cond is not None:
                reference_unet_context = torch.cat([product_proj_embed_un_cond, product_proj_embed_cond])
            else:
                temp_null_prompt = [null_prompt if null_prompt else ""] * effective_batch_size
                null_prompt_text_embeds_for_ref, _ = self._encode_prompt(
                    temp_null_prompt, device, 1, False, None
                )
                current_negative_prompt_embeds = negative_prompt_embeds
                if current_negative_prompt_embeds.shape[0] != null_prompt_text_embeds_for_ref.shape[0]:
                    if current_negative_prompt_embeds.shape[0] == batch_size and null_prompt_text_embeds_for_ref.shape[
                        0] == effective_batch_size:
                        current_negative_prompt_embeds = current_negative_prompt_embeds.repeat_interleave(
                            num_images_per_prompt, dim=0)
                    elif current_negative_prompt_embeds.shape[0] == 0 and null_prompt_text_embeds_for_ref.shape[0] > 0:
                        current_negative_prompt_embeds = torch.zeros_like(null_prompt_text_embeds_for_ref)

                if current_negative_prompt_embeds.shape[0] == null_prompt_text_embeds_for_ref.shape[0]:
                    reference_unet_context = torch.cat(
                        [current_negative_prompt_embeds, null_prompt_text_embeds_for_ref])
                else:
                    logger.warning("Mismatch in reference_unet_context shapes, using zeros for un_cond part.")
                    reference_unet_context = torch.cat(
                        [torch.zeros_like(null_prompt_text_embeds_for_ref), null_prompt_text_embeds_for_ref])

        elif product_proj_embed_cond is not None:
            reference_unet_context = product_proj_embed_cond
        else:
            temp_null_prompt = [null_prompt if null_prompt else ""] * effective_batch_size
            null_prompt_text_embeds_for_ref, _ = self._encode_prompt(temp_null_prompt, device, 1, False, None)
            reference_unet_context = null_prompt_text_embeds_for_ref

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        if self.unet.config.in_channels == 5:
            num_actual_image_channels = 4
        else:
            num_actual_image_channels = self.unet.config.in_channels

        latents = self.prepare_latents(
            effective_batch_size, num_actual_image_channels, height, width,
            prompt_embeds.dtype, device, generator, latents
        )
        current_ref_image = ref_image.to(device=device, dtype=prompt_embeds.dtype)
        if current_ref_image.shape[0] != effective_batch_size:
            if current_ref_image.shape[0] == 1:
                current_ref_image = current_ref_image.repeat(effective_batch_size, 1, 1, 1)
            else:
                raise ValueError(f"ref_image batch size mismatch.")
        if current_ref_image.shape[1] == 3:  # RGB image
            ref_image_for_vae = current_ref_image.to(dtype=self.vae.dtype)
            ref_image_latents_dist = self.vae.encode(ref_image_for_vae).latent_dist
            ref_image_latents = ref_image_latents_dist.sample(generator=generator)
            ref_image_latents = ref_image_latents * self.vae.config.scaling_factor
        elif current_ref_image.shape[1] == self.vae.config.latent_channels:
            ref_image_latents = current_ref_image
        else:
            raise ValueError(f"ref_image has unexpected channel size: {current_ref_image.shape[1]}")

        ref_image_latents = ref_image_latents.to(dtype=latents.dtype)
        latent_h, latent_w = latents.shape[2], latents.shape[3]
        if mask_image_tensor is not None:
            mask_image_tensor_processed = mask_image_tensor.to(device=device, dtype=latents.dtype)
            if mask_image_tensor_processed.shape[0] != effective_batch_size:
                if mask_image_tensor_processed.shape[0] == 1:
                    mask_image_tensor_processed = mask_image_tensor_processed.repeat(effective_batch_size, 1, 1, 1)
                else:
                    raise ValueError(f"mask_image_tensor batch size mismatch.")
            if mask_image_tensor_processed.shape[1] != 1:
                raise ValueError(f"mask_image_tensor should have 1 channel, got {mask_image_tensor_processed.shape[1]}")

            actual_mask_latent_for_unet = torch.nn.functional.interpolate(
                mask_image_tensor_processed, size=(latent_h, latent_w), mode="nearest"
            )
        else:
            actual_mask_latent_for_unet = torch.zeros(
                (effective_batch_size, 1, latent_h, latent_w), device=device, dtype=latents.dtype
            )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        sa_hidden_states_cache = {}

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i == 0:
                    if reference_unet_context is not None and self.reference_unet is not None:
                        ref_unet_batch_size_expected = reference_unet_context.shape[0]
                        current_ref_image_latents_for_ref_unet = ref_image_latents.clone()

                        if current_ref_image_latents_for_ref_unet.shape[0] != ref_unet_batch_size_expected:
                            if do_classifier_free_guidance and current_ref_image_latents_for_ref_unet.shape[
                                0] * 2 == ref_unet_batch_size_expected:
                                current_ref_image_latents_for_ref_unet = current_ref_image_latents_for_ref_unet.repeat(2,
                                                                                                                     1,
                                                                                                                     1,
                                                                                                                     1)
                            elif ref_unet_batch_size_expected % current_ref_image_latents_for_ref_unet.shape[0] == 0:
                                repeats = ref_unet_batch_size_expected // current_ref_image_latents_for_ref_unet.shape[0]
                                current_ref_image_latents_for_ref_unet = current_ref_image_latents_for_ref_unet.repeat(
                                    repeats, 1, 1, 1)
                            else:
                                raise ValueError(
                                    f"Cannot reconcile batch sizes for reference_unet input latents and context.")

                        ref_unet_timesteps = torch.zeros_like(t).expand(current_ref_image_latents_for_ref_unet.shape[0])

                        _ = self.reference_unet(
                            current_ref_image_latents_for_ref_unet,
                            ref_unet_timesteps,
                            encoder_hidden_states=reference_unet_context,
                            return_dict=False,
                        )
                        for name, processor in self.reference_unet.attn_processors.items():
                            if isinstance(processor, HiddenStateCacheAttnProcessor) and hasattr(processor,
                                                                                                'cache') and "hidden_states" in processor.cache:
                                sa_hidden_states_cache[name] = processor.cache["hidden_states"]

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                mask_for_unet_input = torch.cat(
                    [actual_mask_latent_for_unet] * 2) if do_classifier_free_guidance else actual_mask_latent_for_unet

                if self.unet.config.in_channels == 5:
                    unet_input_5ch = torch.cat([latent_model_input, mask_for_unet_input], dim=1)
                else:
                    unet_input_5ch = latent_model_input

                scaled_unet_input = self.scheduler.scale_model_input(unet_input_5ch, t)

                current_unet_cross_attn_kwargs = {}
                if sa_hidden_states_cache:
                    current_unet_cross_attn_kwargs["sa_hidden_states"] = sa_hidden_states_cache

                if reference_item_attention_mask is not None:
                    if do_classifier_free_guidance and reference_item_attention_mask.shape[0] == effective_batch_size:
                        current_unet_cross_attn_kwargs["reference_item_attention_mask"] = reference_item_attention_mask
                    elif not do_classifier_free_guidance:
                        current_unet_cross_attn_kwargs["reference_item_attention_mask"] = reference_item_attention_mask

                if cross_attention_kwargs:
                    current_unet_cross_attn_kwargs.update(cross_attention_kwargs)

                # Predict noise
                if do_classifier_free_guidance:
                    unet_context = torch.cat([negative_prompt_embeds, prompt_embeds])
                    noise_pre_out = self.unet(
                        scaled_unet_input, t,
                        encoder_hidden_states=unet_context,
                        cross_attention_kwargs=current_unet_cross_attn_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pre_un_cond, noise_pre_text = noise_pre_out.chunk(2)
                else:
                    noise_pre_text = self.unet(
                        scaled_unet_input, t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=current_unet_cross_attn_kwargs,
                        return_dict=False,
                    )[0]

                if noise_pre_text.shape[1] != num_actual_image_channels:
                    raise ValueError(
                        f"UNet output noise_pre should have {num_actual_image_channels} channels, but got {noise_pre_text.shape[1]}")

                if do_classifier_free_guidance:
                    noise_pre = noise_pre_un_cond + guidance_scale * (noise_pre_text - noise_pre_un_cond)
                else:
                    noise_pre = noise_pre_text

                latents = self.scheduler.step(noise_pre, t, latents, **extra_step_kwargs).prev_sample

                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
                progress_bar.update()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            if isinstance(has_nsfw_concept, torch.Tensor): has_nsfw_concept = has_nsfw_concept.tolist()
            if not isinstance(has_nsfw_concept, list): has_nsfw_concept = [has_nsfw_concept] * image.shape[0]
            do_denormalize = [not nsfw for nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        if not return_dict:
            return image, has_nsfw_concept

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
