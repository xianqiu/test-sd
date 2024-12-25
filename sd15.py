from diffusers import StableDiffusionPipeline
import torch

from sdxl import SDXLModel


class SD15Model(SDXLModel):

    _config = {
        "directories": {
            "model": "D:/ComfyUI/ComfyUI/models/checkpoints",
            "lora": "D:/ComfyUI/ComfyUI/models/loras",
            "vae": "D:/ComfyUI/ComfyUI/models/vae",
            "save": "outputs"
        },
        "extensions": [".safetensors"],
        "inference": {
            "width": 1024,
            "height": 1536,
            "guidance_scale": 5,
            "num_inference_steps": 20,
            "num_images_per_prompt": 1,
        }
    }

    def __init__(self, model_name, inference_config=None):
        super().__init__(model_name, inference_config)

    def _load_model(self):
        model = StableDiffusionPipeline.from_single_file(
            str(self._model_path),
            torch_dtype=torch.float16,
        )
        return model

