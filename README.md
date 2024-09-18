# Character Generation and Outfit Application using Dreambooth, LoRA, and IP Adapter

## Problem Statement
The objective of this project is to consistently generate a custom character (referred to as "sks character") and apply various outfits using LoRA fine-tuning and IP Adapter. The challenge lies in maintaining the character's identity across different poses and scenes while allowing outfit customization. The project follows the following structure:

1. Initial Character Generation
2. Dreambooth LoRA Finetuning
3. Character Generation with IP-Adapter
4. Advantages and Disadvantages
5. Future Directions

## Initial Character Generation
The initial character generation was performed using **Stable Diffusion**, specifically SDXL, with a custom text prompt for generating the character in a modern cartoon style. I attempted many other methods for controlling the output such as an IP-Adapter for the face and ControlNet for introducing new poses, however I found that I was able to generate similar looking characters through detailed prompt engineering and trial and error. This method is far from optimal in generating consistent characters for finetuning, and more work can be put into this area, however I was capable of generating some suitable initial images:

<div style="display: flex; justify-content: space-between;">
    <img src="/character_images/image_1.png" alt="Image 1" width="300"/>
    <img src="/character_images/image_2.png" alt="Image 2" width="300"/>
    <img src="/character_images/image_7.png" alt="Image 3" width="300"/>
    <img src="/character_images/image_8.png" alt="Image 4" width="300"/>
</div>


## Dreambooth LoRA Finetuning
To maintain consistency across multiple image generations, **Dreambooth and LoRA finetuning** on a pre-trained model was used, resulting in a unique "sks character" style. I chose Dreambooth and LoRA because they the ability to generate high quality unique poses and facial expressions of a character seen in only a few instances with minimal training due to the Low Rank Adaptation. With this it is possible to generate the initial character with much more ease than previously as all of the unique prompt characteristics have now been encoded into the unique token `sks character`. Here are some images produced after finetuning:


### Example Code for LoRA Finetuning

```python
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
import torch

# Load your LoRA weights
lora_weights = "/content/pytorch_lora_weights.safetensors"
pipe.load_lora_weights(lora_weights)
