# Character Generation and Outfit Application using Dreambooth, LoRA, and IP Adapter

## Problem Statement
The objective of this project is to consistently generate a custom character (referred to as "sks character") and apply various outfits using LoRA fine-tuning and IP Adapter. The challenge lies in maintaining the character's identity across different poses and scenes while allowing outfit customization. The project follows the following structure:

1. Initial Character Generation
2. Dreambooth LoRA Finetuning
3. Character Generation with IP-Adapter
4. Advantages and Disadvantages
5. Future Directions

## Initial Character Generation
The initial character generation was performed using **Stable Diffusion** with a custom text prompt for generating the character in a modern cartoon style. To maintain consistency across multiple image generations, we leveraged **Dreambooth finetuning** on a pre-trained model, resulting in a unique "sks character" style.

## Dreambooth LoRA Finetuning
To further customize the character, we used **LoRA (Low-Rank Adaptation)**, which allows efficient fine-tuning of the Dreambooth model without altering the full network. By finetuning with a custom dataset of character images, LoRA helped keep the character's identity while introducing new outfits.

### Example Code for LoRA Finetuning

```python
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
import torch

# Load your LoRA weights
lora_weights = "/content/pytorch_lora_weights.safetensors"
pipe.load_lora_weights(lora_weights)
