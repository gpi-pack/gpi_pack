'''
Implementation of Generating Images with Stable Diffusion.

Currently, the implementation supports the Stable Diffusion 1.5 and 2.1 architectures,
and it does not suppoert the Stable Diffusion 3 and 3.5 architectures.
We will update the implementation to support the latest architectures in the future.

Package update on 2025-10-15:
- Update support for new checkpoint https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
'''

from __future__ import annotations

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler
from PIL import Image
import os
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

from typing import List, Tuple, Optional, Union


def pad_to_multiple_of_8(img: Image.Image, pad_mode="reflect") -> Image.Image:
    h, w = img.height, img.width
    pad_h = (8 - h % 8) % 8 # Calculate padding needed to make height a multiple of 8
    pad_w = (8 - w % 8) % 8 # Calculate padding needed to make width a multiple of 8

    if pad_h == 0 and pad_w == 0:
        return img                    # already fine

    print(f"Padding image from ({h}, {w}) to ({h + pad_h}, {w + pad_w}) with {pad_mode} padding.")
    print("This is to ensure that the image dimensions are multiples of 8 for Diffusion Model inputs.")

    # symmetric padding (left/right, top/bottom)
    padding = (
        pad_w // 2,
        pad_h // 2,
        pad_w - pad_w // 2,
        pad_h - pad_h // 2,
    )
    img_np = np.array(img.convert("RGB"))
    img_np = np.pad(
        img_np,
        ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0)),
        mode=pad_mode
    )
    return Image.fromarray(img_np)


class StableDiffusionImg2ImgExtractor:
    '''
    A class to generate images from text prompts and image inputs using Stable Diffusion and extract hidden states.
    This class is based on the Stable Diffusion architecture (version 1.5 and 2.1) and uses components like VAE, UNet, and text encoders.
    '''
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = None,
        cache_dir: Optional[str] = None,
        dtype: torch.dtype = torch.float32
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print("Device:", self.device)
        self.cache_dir = cache_dir
        self.dtype = dtype

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

        # Load and configure all model components
        ## Component 1: VAE (Variational Autoencoder)
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            cache_dir=cache_dir
        ).to(self.device).to(dtype)

        ## Component 2: Tokenizer and Text Encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
            cache_dir=cache_dir
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            cache_dir=cache_dir
        ).to(self.device).to(dtype)

        ## Component 3: UNet (for image generation)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            cache_dir=cache_dir
        ).to(self.device).to(dtype)

        ## Component 4: Scheduler (for denoising steps)
        self.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            cache_dir=cache_dir
        )

        # Set all components to evaluation mode
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()

    def preprocess_image(self, image: Union[Image.Image, str], max_size: int = 512) -> torch.Tensor:
        """
        Preprocess input image for the model.

        - Convert to RGB (ensures 3 channels)
        - Convert to tensor
        - Normalize using 3-channel means and std
        - Optionally pad to ensure dimensions are multiples of 8
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            image = image.resize((new_width, new_height), Image.LANCZOS)

        image = pad_to_multiple_of_8(image, pad_mode="reflect")
        image = image.convert("RGB")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        image = transform(image).unsqueeze(0).to(self.device).to(self.dtype)

        return image

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[str] = None
    ) -> torch.Tensor:
        """
        Encode the prompt and negative prompt into text embeddings.

        Args:
            prompt: Text prompt or list of prompts to encode
            negative_prompt: Optional negative prompt to encode (negative prompt is the prompt that the model should not generate)

        """
        if isinstance(prompt, str):
            prompt = [prompt]

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        text_embeddings = self.text_encoder(
            text_input.input_ids.to(self.device)
        )[0]

        if negative_prompt is None:
            uncond_input = self.tokenizer(
                [""] * len(prompt),
                padding="max_length",
                max_length=text_input.input_ids.shape[-1],
                return_tensors="pt"
            )
        else:
            uncond_input = self.tokenizer(
                [negative_prompt] * len(prompt),
                padding="max_length",
                max_length=text_input.input_ids.shape[-1],
                return_tensors="pt"
            )

        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.device)
        )[0]

        return torch.cat([uncond_embeddings, text_embeddings])

    def get_hidden_states(
        self,
        input_image: Union[Image.Image, str],
        prompt: str,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Generate hidden states and latents from input image and prompt.

        Args:
            input_image: Input image to transform
            prompt: Text prompt for transformation
            strength: Strength of the transformation (0-1)
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            negative_prompt: Optional negative prompt
            seed: Random seed for reproducibility

        Returns:
            Tuple of (hidden states list, final latents)
        """
        # to ensure reproducibility of the generated latents
        if seed is not None:
            torch.manual_seed(seed)

        # Encode input image
        init_image = self.preprocess_image(input_image)
        with torch.no_grad():
            init_latents = self.vae.encode(init_image).latent_dist.sample()
            init_latents = 0.18215 * init_latents

        # Get text embeddings
        text_embeddings = self.encode_prompt(prompt, negative_prompt)

        if strength == 0:
            # If strength is 0, return the initial latents directly
            return init_latents

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        offset = self.scheduler.config.get("steps_offset", 0)
        init_timestep = int(num_inference_steps * strength) + offset
        init_timestep = min(init_timestep, num_inference_steps)
        timesteps = self.scheduler.timesteps[-init_timestep]
        timesteps = torch.tensor([timesteps], device=self.device)

        # Add noise to latents
        noise = torch.randn_like(init_latents)
        init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

        latents = init_latents
        t_start = max(num_inference_steps - init_timestep + offset, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        # Denoising loop
        for t in timesteps:
            # Expand latents for classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Get UNet predictions
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images."""
        # Scale back the latents
        latents = 1 / 0.18215 * latents

        # Decode latents using VAE
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        return image

    def transform_image(
        self,
        input_image: Union[Image.Image, str],
        prompt: str,
        strength: float = 0.8,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
        return_hidden_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]]:
        """
        Transform input image according to prompt.

        Args:
            input_image: Input image to transform
            prompt: Text prompt for transformation
            strength: Strength of the transformation (0-1)
            negative_prompt: Optional negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            seed: Random seed for reproducibility
            show: Whether to display the generated image
            save_path: Optional path to save the image
            return_hidden_states: Whether to return hidden states

        Returns:
            Transformed image tensor or tuple of (image, hidden states, latents)
        """
        latents = self.get_hidden_states(
            input_image,
            prompt,
            strength,
            num_inference_steps,
            guidance_scale,
            negative_prompt,
            seed
        )
        image = self.decode_latents(latents)
        # if save_path is provided, save the image in the specified path
        if save_path:
            self.save_image(image, save_path)
        # If return_hidden_states is True, return the image and latents
        if return_hidden_states:
            return image, latents
        # Otherwise, just return the image
        return image

    def save_image(self, image: torch.Tensor, save_path: str) -> None:
        """Save the generated image to a file."""
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).round().astype("uint8")
        pil_image = Image.fromarray(image)
        pil_image.save(save_path)
        print(f"Image saved to {save_path}")


def extract_images(
        images: Union[Image.Image, str, List[Union[Image.Image, str]]],
        prompts: Union[str, List[str]],
        output_hidden_dir: str,
        output_image_dir: Optional[str] = None,
        save_name: str = "gen",
        prefix_hidden: str = "hidden",
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        strength: float = 0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ):
    '''
    Generate images from input images and prompts, and extract hidden states.

    Args:
        images: Input images (single image or list of images)
        prompts: Text prompts (single prompt or list of prompts)
        output_hidden_dir: Directory to save hidden states
        output_image_dir: Directory to save generated images (optional)
        save_name: Base name for saved files
        prefix_hidden: Prefix for hidden state files
        model_id: Model ID for Stable Diffusion
        device: Device to run the model on (default is "cuda" if available)
        cache_dir: Directory to cache model files
        strength: Strength of the transformation (0-1)
        num_inference_steps: Number of denoising steps
        guidance_scale: Scale for classifier-free guidance
        negative_prompt: Optional negative prompt
        seed: Random seed for reproducibility
    '''
    if isinstance(images, (Image.Image, str)):
        images = [images]
    if isinstance(prompts, str):
        prompts = [prompts]

    if len(images) != len(prompts):
        raise ValueError(
            f"Number of images ({len(images)}) and prompts "
            f"({len(prompts)}) must match."
        )

    Path(output_hidden_dir).mkdir(parents=True, exist_ok=True)
    if output_image_dir:
        Path(output_image_dir).mkdir(parents=True, exist_ok=True)

    extractor = StableDiffusionImg2ImgExtractor(
        model_id=model_id,
        device=device,
        cache_dir=cache_dir,
    )

    all_latents = []

    for i, (img_in, prompt) in enumerate(
        tqdm(zip(images, prompts), total=len(images), desc="Processing")
    ):
        gen_img, latents = extractor.transform_image(
            input_image=img_in,
            prompt=prompt,
            strength=strength,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            return_hidden_states=True,
        )

        if output_image_dir:
            out_img_path = os.path.join(output_image_dir, f"{save_name}_{i}.png")
            extractor.save_image(gen_img, out_img_path)

        out_latent_path = os.path.join(output_hidden_dir, f"{prefix_hidden}_{i}.pt")
        torch.save(latents, out_latent_path)
        all_latents.append(latents)