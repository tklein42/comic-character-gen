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

<div style="text-align: center;">
    <h3 style="font-weight: normal;">Outfits</h3>
    <div style="display: flex; justify-content: space-between; gap: 10px;">
        <img src="/outfits/outfit_1.png" alt="Image 1" width="200"/>
        <img src="/outfits/outfit_1.png" alt="Image 2" width="200"/>
    </div>
</div>

In using these the following results were gathered:

<div style="text-align: center;">
    <h3 style="font-weight: normal;">Set 1 - Generated From Outfit 1</h3>
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

## Observations
From these results we can see that the model was able to generalize our character in different poses and expressions while maintaining a similar outfit throughout. However there are major drawbacks that need to be addressed.

1. Altered Character Style

As seen throughout the generated images we can see that in the process of fusing the image and text prompts the resulting 'sks character' loses some of its characteristics from the initial training set. Where the characteristics, although still not realistic, moving in the direction of realism, whereas the original model is capable of a much more cartoonish appearance.

2. Altered Colored Palette

Another observation that can be seen throughout both sets of images is caused by a similar issue in fusing the two prompts together. This is the issue where the colors used within the outfit image are taken as a palette for image generation. As seen by set 1 having more cooler colors present and set 2 having more warmer colors present. This is likely due to the high weight placed into the style component of `ip-adapter-scale` which was shown to be necessary for applying the outfit using this method.

## Future Directions
From the two previously mentioned issues I began working on a secondary method incorporating a inpainting with the SDXL model, as two 'paint' the desired outfit onto my character. However this technique involves first generating a mask for the model to be directed to the desired area to fill. Since the desire of this project is it to be simple for other users to test it is necessary to incorporate an automatic method to generate the desired masks. To accomplish this I began by implementing a pipeline known as Grounded SAM which incorporates two separate models. The first being Grounded DINO which is to be used as an object detector for identifying things of interest such as a `body` or `clothing` and then having the corresponding bounding boxes used for the next step. This next step is known as SAM, or the Segment Anything Model, which can take a region of interest from an initial image and perform a segmentation. This segmentation will then be used as a mask for the inpainting. I set this up by doing the following:

#### Initialization
```python
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForMaskGeneration

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"

# Define Grounded DINO and processor
detector_processor = AutoProcessor.from_pretrained(model_id)
detector = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# Define the SAM model and processor
segmenter_id = "facebook/sam-vit-huge"
segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to("cuda")
segmentator_processor = AutoProcessor.from_pretrained(segmenter_id)
```

#### Inference 

##### Bounding Box Generation
```python

text = "clothing."

# Use Grounded DINO to extract bounding boxes
inputs = detector_processor(images=initial_img, text=text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = detector(**inputs)

results = detector_processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[initial_img.size[::-1]]
)
```

##### Mask Generation
```python
# Use SAM with the extracted bounding boxes
inputs_sam = segmentator_processor(images=initial_img, input_boxes=[boxes], return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs_sam = segmentator(**inputs_sam)

mask = outputs_sam.pred_masks.squeeze().cpu().numpy()
```

Further explanations for how these are used in the alternate pipeline, that is still under development, can be found in the repository under `alternate_pipeline.ipynb`. However once these masks are generated the same SDXL model can be used in order to generate the character in the desired outfit. Here are some initial results:

<div style="text-align: center; display: flex; justify-content: space-between;">
    <h3 style="font-weight: normal;">Inpainting Results From Outfit 1</h3>
    <div style="display: flex; justify-content: space-between; gap: 10px;">
        <img src="/alternate_pipeline_images/pipe2-outfit1-scene1.png" alt="Image 1" width="150"/>
        <img src="/alternate_pipeline_images/pipe2-outfit1-scene2.png" alt="Image 2" width="150"/>
        <img src="/alternate_pipeline_images/pipe2-outfit1-scene3.png" alt="Image 3" width="150"/>
    </div>
</div>

#### Observations
From these results we can see that the model was able to generalize our character in different poses and expressions while maintaining a similar outfit and even be able to produce much more diverse color palettes and backgrounds. However this method is showing some major drawbacks and possibly some other oversights that should be addressed before continuation of project.

1. Visual Artifacts

As seen throughout each of the images there is always visual artifacts from the inpainting as the background hasn't seamless been generated, whether this be an issue with the initial mask or the SDXL model itself. Inorder to minimize this it was assumed that a simple solution would be to reduce the denoising strength as to keep the image as close to the original as possible while only updating the outfit, however this helped identify the following oversight.

2. Unique Character Lacking Variability

From my experiments in generating the character using inpainting I found that it was extremely difficult to have my `sks character` generate anything outside of a jacket, and simply put, I failed to recognize this sooner as now although the model recognizes the features of the character it also identifies the jacket as a key characteristic. Because of this when imploying the inpainting, the model was capable of producing similar colors, such as a black jacket and jeans for the case with outfit 1, however it was struggling to match the outfit exactly with only the t-shirt.
