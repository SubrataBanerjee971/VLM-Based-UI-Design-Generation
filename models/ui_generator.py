"""
ui_generator.py
───────────────
UI Image Generator:
    Stable Diffusion v1-5 + ControlNet (scribble) conditioned on:
        • Sketch image       (ControlNet spatial conditioning)
        • CLIP text embedding (cross-attention in UNet)
        • Fused h_ui vector  (injected as extra conditioning token)

Architecture flow:
    sketch → ControlNet → residuals
    prompt → CLIP text encoder → UNet cross-attention
    h_ui   → injected as learnable conditioning token alongside CLIP embeds

This file defines:
    UIGeneratorModel  – the full generation model used at inference
    UIGeneratorUNet   – thin wrapper that injects h_ui into the UNet
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDPMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
#  Conditioning Injector
# ──────────────────────────────────────────────────────────────────────────────

class FusedConditioningInjector(nn.Module):
    """
    Projects h_ui (B, d_vlm) → (B, 1, d_text) and prepends it to the
    CLIP text embeddings so the UNet cross-attention sees it as an extra
    conditioning token.
    """

    def __init__(self, d_vlm: int = 512, d_text: int = 768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_vlm, d_text),
            nn.LayerNorm(d_text),
            nn.GELU(),
            nn.Linear(d_text, d_text),
        )

    def forward(
        self,
        h_ui: torch.Tensor,          # (B, d_vlm)
        text_embeds: torch.Tensor,   # (B, L, d_text)
    ) -> torch.Tensor:
        """Returns (B, L+1, d_text)."""
        extra = self.proj(h_ui).unsqueeze(1)              # (B, 1, d_text)
        return torch.cat([extra, text_embeds], dim=1)     # (B, L+1, d_text)


# ──────────────────────────────────────────────────────────────────────────────
#  Full UI Generator
# ──────────────────────────────────────────────────────────────────────────────

class UIGeneratorModel(nn.Module):
    """
    Wraps the Stable Diffusion + ControlNet pipeline and adds:
        1. FusedConditioningInjector   – injects h_ui into text embeds
        2. Fine-tunable UNet cross-attn layers via LoRA (optional)
    """

    def __init__(
        self,
        base_model: str         = "runwayml/stable-diffusion-v1-5",
        controlnet_model: str   = "lllyasviel/sd-controlnet-scribble",
        d_vlm: int              = 512,
        noise_scheduler: str    = "DDPM",
        num_train_timesteps: int = 1000,
        device: str             = "cuda",
    ):
        super().__init__()
        self.device_str = device
        # Use float32 on CPU (always safe), float16 only on real CUDA
        _use_fp16 = (device == "cuda" and torch.cuda.is_available())
        _dtype = torch.float16 if _use_fp16 else torch.float32

        # ── ControlNet ───────────────────────────────────────────────────────
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=_dtype,
        )

        # ── UNet ─────────────────────────────────────────────────────────────
        self.unet = UNet2DConditionModel.from_pretrained(
            base_model, subfolder="unet", torch_dtype=_dtype,
        )

        # ── VAE ──────────────────────────────────────────────────────────────
        self.vae = AutoencoderKL.from_pretrained(
            base_model, subfolder="vae", torch_dtype=_dtype,
        )

        # ── Text encoder ─────────────────────────────────────────────────────
        self.text_encoder = CLIPTextModel.from_pretrained(
            base_model,
            subfolder="text_encoder",
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            base_model,
            subfolder="tokenizer",
        )

        # ── Noise scheduler ──────────────────────────────────────────────────
        sched_kwargs = dict(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=num_train_timesteps,
        )
        self.noise_scheduler = (
            DDPMScheduler(**sched_kwargs)
            if noise_scheduler == "DDPM"
            else DDIMScheduler(**sched_kwargs)
        )

        # ── Fused conditioning injector ──────────────────────────────────────
        d_text = self.text_encoder.config.hidden_size
        self.injector = FusedConditioningInjector(d_vlm=d_vlm, d_text=d_text)

        # Freeze VAE (not trained)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def encode_prompt(
        self,
        prompts: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """Tokenise and encode a batch of prompts → (B, L, d_text)."""
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            enc = self.text_encoder(**tokens)
        return enc.last_hidden_state   # (B, L, d_text)

    def encode_images_to_latents(
        self,
        images: torch.Tensor,   # (B, 3, H, W) in [-1, 1]
    ) -> torch.Tensor:
        """Encode images to VAE latent space."""
        # Cast to VAE's dtype to prevent float/half mismatches on CPU
        images = images.to(dtype=next(self.vae.parameters()).dtype)
        with torch.no_grad():
            dist = self.vae.encode(images).latent_dist
        return dist.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents → pixel images in [-1, 1]."""
        latents = latents / self.vae.config.scaling_factor
        return self.vae.decode(latents).sample

    # ── Training forward pass ─────────────────────────────────────────────────

    def forward(
        self,
        target_images: torch.Tensor,   # (B, 3, H, W) in [-1,1]
        sketch_images: torch.Tensor,   # (B, 3, H, W) in [-1,1] – ControlNet input
        prompts: list[str],
        h_ui: torch.Tensor,            # (B, d_vlm) fused embedding from CrossAttn
    ) -> torch.Tensor:
        """
        Compute diffusion training loss.

        Returns:
            loss : scalar MSE between predicted and actual noise
        """
        device = target_images.device
        # Infer model dtype from UNet weights (float32 on CPU, float16 on CUDA)
        model_dtype = next(self.unet.parameters()).dtype

        # 1. Encode target → latents
        latents = self.encode_images_to_latents(target_images)   # dtype matches VAE

        # 2. Sample noise + timesteps
        noise = torch.randn_like(latents)
        B = latents.size(0)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device, dtype=torch.long,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(dtype=model_dtype)

        # 3. Build conditioning embeddings
        text_embeds = self.encode_prompt(prompts, device)    # (B, L, d_text)
        encoder_hidden_states = self.injector(h_ui.to(text_embeds.dtype), text_embeds)
        encoder_hidden_states = encoder_hidden_states.to(dtype=model_dtype)

        # Cast sketch to model dtype for ControlNet
        sketch_cond = sketch_images.to(dtype=model_dtype)

        # 4. ControlNet forward
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=sketch_cond,
            return_dict=False,
        )

        # 5. UNet forward
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample

        # 6. MSE loss on noise prediction (always float32 for numerical stability)
        loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
        return loss


    # ── Inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        sketch_image: Image.Image | torch.Tensor,
        prompt: str,
        h_ui: torch.Tensor | None,
        num_inference_steps: int = 50,
        guidance_scale: float   = 7.5,
        controlnet_scale: float = 1.0,
        output_size: int        = 512,
        device: str             = "cuda",
    ) -> Image.Image:
        """
        Generate a UI design image from a sketch + prompt.

        Args:
            sketch_image : hand-drawn sketch as PIL Image or tensor
            prompt       : text description of desired UI
            h_ui         : fused multimodal embedding; None → zeros
            …

        Returns:
            PIL.Image of the generated UI
        """
        from data.preprocessor import pil_to_tensor, tensor_to_pil
        from diffusers import StableDiffusionControlNetPipeline

        # Build a temporary pipeline for inference
        pipe = StableDiffusionControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=DDIMScheduler.from_config(
                self.noise_scheduler.config
            ),
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
        ).to(device)

        # Prepare sketch tensor
        if isinstance(sketch_image, Image.Image):
            sketch_tensor = pil_to_tensor(sketch_image).unsqueeze(0).to(device)
        else:
            sketch_tensor = sketch_image.to(device)
        # ControlNet expects [0,1] float
        sketch_01 = (sketch_tensor * 0.5 + 0.5).clamp(0, 1)

        # Prepare conditioning
        d_text = self.text_encoder.config.hidden_size
        if h_ui is None:
            h_ui = torch.zeros(1, self.injector.proj[0].in_features, device=device)

        text_embeds = self.encode_prompt([prompt], torch.device(device))
        encoder_hidden_states = self.injector(h_ui.to(text_embeds.dtype), text_embeds)

        # Use unconditional embeds for classifier-free guidance
        uncond_tokens = self.tokenizer(
            [""], padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)
        uncond_embeds = self.text_encoder(**uncond_tokens).last_hidden_state
        # Pad uncond to match fused length
        extra_zeros = torch.zeros(
            1, encoder_hidden_states.size(1) - uncond_embeds.size(1), d_text,
            device=device, dtype=uncond_embeds.dtype
        )
        uncond_embeds = torch.cat([extra_zeros, uncond_embeds], dim=1)

        prompt_embeds = torch.cat([uncond_embeds, encoder_hidden_states])

        # Denoising loop
        self.noise_scheduler.set_timesteps(num_inference_steps)
        latents = torch.randn(1, 4, output_size // 8, output_size // 8, device=device)

        for t in self.noise_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)

            down_res, mid_res = self.controlnet(
                latent_model_input, t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=torch.cat([sketch_01] * 2),
                conditioning_scale=controlnet_scale,
                return_dict=False,
            )

            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        pixels = self.decode_latents(latents)          # (1,3,H,W) in [-1,1]
        return tensor_to_pil(pixels)
