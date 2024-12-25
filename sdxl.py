import time
import re
from pathlib import Path

from diffusers import StableDiffusionXLPipeline, AutoPipelineForImage2Image
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
import torch
from PIL import Image


class SDXLModel(object):

    _config = {
        "directories": {
            "model": "D:/ComfyUI/ComfyUI/models/checkpoints",
            "lora": "D:/ComfyUI/ComfyUI/models/loras",
            "save": "outputs"
        },
        "extensions": [".safetensors"],
        "inference": {
            "width": 1024,
            "height": 1536,
            "guidance_scale": 5,
            "num_inference_steps": 20,
            "num_images_per_prompt": 1,
        },
    }

    def __init__(self, model_name, inference_config=None):
        self._model_name = model_name
        self._device = 'cuda'
        self._inference_config = inference_config or {}
        self._model = self._load_model().to(self._device)
        self._set_model_sampler()  # after load_model
        self._update_inference_config()
        self._lora_adapter_weights = {}
        # image to image
        self._model_image = None
        self._refer_image_path = None

    def _update_inference_config(self):
        inference_default = self._config['inference']
        # 缺失的参数，用默认值替代
        missing_items = {key: value for key, value in inference_default.items()
                         if key not in self._inference_config}
        self._inference_config.update(missing_items)

    @property
    def _model_path(self):
        directory = Path(self._config['directories']['model'])
        for extension in self._config['extensions']:
            filepath = directory / f"{self._model_name}{extension}"
            if filepath.exists():
                return filepath
        return None

    def _get_lora_path(self, name):
        directory = Path(self._config['directories']['lora'])
        for extension in self._config['extensions']:
            filepath = directory / f"{name}{extension}"
            if filepath.exists():
                return filepath
        return None

    def add_lora(self, name, weight):
        filepath = self._get_lora_path(name)
        self._lora_adapter_weights[name] = weight
        self._model.load_lora_weights(filepath, adapter_name=name)
        return self

    def _set_adapters(self):
        if not self._lora_adapter_weights:
            return
        adapter_names = list(self._lora_adapter_weights.keys())
        weights = list(self._lora_adapter_weights.values())
        self._model.set_adapters(adapter_names, adapter_weights=weights)

    def _load_model(self):
        model = StableDiffusionXLPipeline.from_single_file(
            self._model_path,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        return model

    def _set_model_sampler(self):
        sampler = self._inference_config.get('sampler')
        if sampler == "EulerA":
            self._model.scheduler = EulerAncestralDiscreteScheduler.from_config(self._model.scheduler.config)
        elif sampler == "DPM++ 2M Karras":
            self._model.scheduler = DPMSolverMultistepScheduler.from_config(self._model.scheduler.config)

    def _add_prompt_prefix(self, prompt=None, negative_prompt=None):

        def concat(prefix=None, words=None):
            if prefix and words:
                return ",".join([prefix.strip(), words.strip()])
            else:
                either = prefix or words
                if either:
                    return either.strip()
            return None

        prompt_prefix = self._inference_config.get("prompt_prefix")
        prompt = concat(prompt_prefix, prompt)
        negative_prefix = self._inference_config.get("negative_prompt_prefix")
        negative_prompt = concat(negative_prefix, negative_prompt)

        return prompt, negative_prompt

    def _save_image(self, images):
        prefix = re.sub(r'[.\-_]', '', self._model_name.lower())[0: 6]
        for index, image in enumerate(images):
            filename = f"{prefix}{int(time.time())}_{index}"
            image.save(f"{self._config['directories']['save']}/{filename}.png")

    def _print_inference_params(self, params):
        print("\n==== Inference Parameters ====")

        print(f"|-- model_name = {self._model_name}")
        mode = "Image to Image" if self._refer_image_path else "Text to Image"
        print(f"|-- mode = {mode}")

        if self._refer_image_path:
            print(f"|-- refer_image_path = {self._refer_image_path}")

        # lora
        for lora, weight in self._lora_adapter_weights.items():
            print(f"|-- lora_name = {lora}, weight = {weight}")

        # inference parameters
        for key, value in params.items():
            print(f"|-- {key} = {value}")

        print("==== END ====\n")

    def add_refer_image(self, image_path, strength=0.5):
        self._refer_image_path = Path(image_path)
        assert self._refer_image_path.exists(), FileNotFoundError(self._refer_image_path)
        self._inference_config["image"] = Image.open(self._refer_image_path)
        self._inference_config["strength"] = strength
        self._model = AutoPipelineForImage2Image.from_pipe(self._model).to(self._device)

    def _get_inference_kwargs(self):
        ignore_keys = ["prompt_prefix", "negative_prompt_prefix", "vae", "sampler"]
        inference_kwargs = {key: value for key, value in self._inference_config.items()
                            if key not in ignore_keys}
        return inference_kwargs

    def generate(self, prompt=None, negative_prompt=None):
        # apply lora if selected
        self._set_adapters()

        # format inference kwargs
        inference_kwargs = self._get_inference_kwargs()
        prompt, negative_prompt = self._add_prompt_prefix(prompt, negative_prompt)
        inference_kwargs.update({"prompt": prompt, "negative_prompt": negative_prompt})
        # print info
        self._print_inference_params(inference_kwargs)

        # inference
        images = self._model(**inference_kwargs)["images"]
        # save images
        self._save_image(images)


class SDModelWrap(object):

    _inference_config = {
        "prompt_prefix": "",
        "negative_prompt_prefix": "",
        "width": 1024,
        "height": 1360,
        "num_inference_steps": 30,
        "guidance_scale": 7,
        "num_images_per_prompt": 1,
    }

    def __init__(self):
        self._model_name = None
        self._model = None

    def text_to_image(self, prompt, negative_prompt=None):
        self._model.generate(prompt, negative_prompt)

    def image_to_image(self, image_path, strength=0.5,
                       prompt=None, negative_prompt=None):
        self._model.add_refer_image(image_path, strength)
        self._model.generate(prompt, negative_prompt)
