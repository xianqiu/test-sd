from sdxl import SDXLModel, SDModelWrap


class ZenIllu(SDModelWrap):

    # Base Model: SDXL

    _inference_config = {
        "prompt_prefix": "(masterpiece), best, 8k wallpapers, illustrations",
        "negative_prompt_prefix": """nsfw,(((pubic))), ((((pubic_hair)))), sketch, duplicate, ugly, 
        huge eyes, text, logo, monochrome, worst face, (bad and mutated hands:1.3), 
        (worst quality:1.7), (low quality:1.7), (blurry:1.7),horror, geometry, bad_prompt, 
        (bad hands), (missing fingers), multiple limbs, bad anatomy, (interlocked fingers:1.2),(interlocked leg:1.2), 
        Ugly Fingers, (extra digit and hands and fingers and legs and arms:1.4), 
        crown braid, (deformed fingers:1.2), (long fingers:1.2),succubus wings,horn,succubus horn,
        succubus hairstyle, (bad-artist-anime), bad-artist, bad hand""",
        "num_inference_steps": 25,
        "width": 1024,
        "height": 1536,
        "guidance_scale": 7,
        "sampler": "DPM++ 2M Karras",
    }

    def __init__(self):
        super().__init__()
        self._lora_name = "Zen_Illustration-SDXL_v1_0"
        self._model_name = "GhostXL_V1.0-Baked_VAE"
        self._model = SDXLModel(self._model_name, self._inference_config)
        self._model.add_lora(self._lora_name, 0.7)


prompt1 = """full body,beautiful eyes, facial contours, beauty Long black hair, 
        (the background is round Chinese elements, red Chinese elements, 
        ancient Chinese costumes, Phoenix, Zen, Realm, Tao, Fantastic)"""


if __name__ == '__main__':

    z = ZenIllu()
    z.text_to_image(prompt1)