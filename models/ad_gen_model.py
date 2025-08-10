import torch

class ADGenModel(torch.nn.Module):
    def __init__(self, unet, ref_unet, proj, adapter_modules) -> None:
        super().__init__()
        self.unet = unet
        self.ref_unet = ref_unet
        self.proj = proj
        self.adapter_modules = adapter_modules

    def forward(self, encoder_hidden_states, latents, ref_latents, clip_image_embeddings, timesteps,
                reference_item_attention_mask=None):
        ref_timesteps = torch.zeros_like(timesteps)
        product_proj_embed = self.proj(clip_image_embeddings)

        _ = self.ref_unet(
            ref_latents,
            ref_timesteps,
            product_proj_embed,
            return_dict=False,
        )
        sa_hidden_states = {}
        for name in self.ref_unet.attn_processors.keys():
            if hasattr(self.ref_unet.attn_processors[name], 'cache') and "hidden_states" in \
                    self.ref_unet.attn_processors[name].cache:
                sa_hidden_states[name] = self.ref_unet.attn_processors[name].cache["hidden_states"]

        cross_attention_kwargs = {"sa_hidden_states": sa_hidden_states}
        if reference_item_attention_mask is not None:
            cross_attention_kwargs["reference_item_attention_mask"] = reference_item_attention_mask

        noise_pre = self.unet(
            latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs
        ).sample
        return noise_pre
