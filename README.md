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
    <img src="/character_images/image_1.png" alt="Image 1" width="200"/>
    <img src="/character_images/image_2.png" alt="Image 2" width="200"/>
    <img src="/character_images/image_7.png" alt="Image 3" width="200"/>
    <img src="/character_images/image_8.png" alt="Image 4" width="200"/>
</div>


## Dreambooth LoRA Finetuning
To maintain consistency across multiple image generations, **Dreambooth and LoRA finetuning** on a pre-trained model was used, resulting in a unique "sks character" style. I chose Dreambooth and LoRA because they the ability to generate high quality unique poses and facial expressions of a character seen in only a few instances with minimal training due to the Low Rank Adaptation. With this it is possible to generate the initial character with much more ease than previously as all of the unique prompt characteristics have now been encoded into the unique token `sks character`. Here are some images produced after finetuning:

<div style="display: flex; justify-content: space-between;">
    <img src="/images_after_finetuning/dreambooth_lora_image_1.png" alt="Image 1" width="200"/>
    <img src="/images_after_finetuning/dreambooth_lora_image_2.png" alt="Image 1" width="200"/>
    <img src="/images_after_finetuning/dreambooth_lora_image_3.png" alt="Image 1" width="200"/>
    <img src="/images_after_finetuning/dreambooth_lora_image_4.png" alt="Image 1" width="200"/>
</div>

## Character Generation with IP-Adapter
With the LoRA weights saved, we simply load the weights back into the SDXL model so that information about the character and corresponding token are now known. This can be accomplished as follows:

```python
from diffusers import StableDiffusionXLPipeline

# Load txt2img pipe
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    )

# Load the LoRA weights
lora_weights_path = "/content/pytorch_lora_weights.safetensors"
txt2img_pipe.load_lora_weights(lora_weights_path)

txt2img_pipe.to("cuda")
```

Once the weights are loading we can use the functionality built into hugging face diffusers to implement the IP-Adapter. The IP-Adapter, effectively adds the ability for the model to take an image prompt. It fuses the image prompt along with the original text prompt to give more control over the appearance of the image. This can be loaded simply by the following code:

```python
# Load IP-Adapter
txt2img_pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

# Image Prompt Scale
scale = {
    "down": {"block_2": [0.0, 0.4]}, # Layout
    "up": {"block_0": [0.0, 0.85, 0.0]}, # Style
}
txt2img_pipe.set_ip_adapter_scale(scale)
```

Using this method we can encode information regarding the desired outfit to reinforce the text prompt. I generated both from scratch as to attempt to not confuse the model with character and  design choice also allowed me to avoid using a segmentation method as to extract only the outfit from a possibly noisy image. The following two outfits were used:

<div style="display: flex; justify-content: space-between; gap: 10px;">
    <figure>
        <img src="/outfits/outfit_1.png" alt="Image 1" width="200"/>
        <figcaption>Outfit 1</figcaption>
    </figure>
    <figure>
        <img src="/outfits/outfit_2.png" alt="Image 2" width="200"/>
        <figcaption>Outfit 2</figcaption>
    </figure>
</div>

In using these the following results were gathered:

<div style="text-align: center;">
    <h3>Set 1 - Generated From Outfit 1</h3>
    <div style="display: flex; justify-content: space-between; gap: 10px;">
        <img src="/set_1/pipe1-outfit1-scene1.png" alt="Image 1" width="150"/>
        <img src="/set_1/pipe1-outfit1-scene2.png" alt="Image 2" width="150"/>
        <img src="/set_1/pipe1-outfit1-scene3.png" alt="Image 3" width="150"/>
        <img src="/set_1/pipe1-outfit1-scene4.png" alt="Image 4" width="150"/>
        <img src="/set_1/pipe1-outfit1-scene5.png" alt="Image 5" width="150"/>
    </div>
</div>

<div style="text-align: center;">
    <h3>Set 2 - Generated From Outfit 2</h3>
    <div style="display: flex; justify-content: space-between; gap: 10px;">
        <img src="/set_2/pipe1-outfit2-scene1.png" alt="Image 1" width="150"/>
        <img src="/set_2/pipe1-outfit2-scene2.png" alt="Image 2" width="150"/>
        <img src="/set_2/pipe1-outfit2-scene3.png" alt="Image 3" width="150"/>
        <img src="/set_2/pipe1-outfit2-scene4.png" alt="Image 4" width="150"/>
        <img src="/set_2/pipe1-outfit2-scene5.png" alt="Image 5" width="150"/>
    </div>
</div>

### Example Code for LoRA Finetuning

```python
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
import torch

# Load your LoRA weights
lora_weights = "/content/pytorch_lora_weights.safetensors"
pipe.load_lora_weights(lora_weights)
