from sd15 import SD15Model
from sdxl import SDModelWrap


class HealingIllu(SDModelWrap):

    # Base Model: SD15

    _inference_config = {
        "prompt_prefix": "Best quality, masterpiece, ultra-detailed",
        "negative_prompt_prefix": """(nsfw:1.5),verybadimagenegative_v1.3,ng_deepnegative_v1_75t,(ugly face:0.8),
        cross-eyed,sketches,(worst quality:2),(low quality:2),(normal quality:2),lowres,normal quality,((monochrome)),
        ((grayscale)),skin spots,acnes,skin blemishes,bad anatomy,nsfw,, lowres, bad anatomy, bad hands, text, error, 
        missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, 
        signature, watermark, username, blurry""",
        "width": 1304,
        "height": 1912,
        "num_inference_steps": 30,
        "guidance_scale": 7,
        "sampler": "EulerA",
    }

    def __init__(self):
        super().__init__()
        self._model_name = "Healing_Illustrations_Anime_v1.0"
        self._model = SD15Model(self._model_name, self._inference_config)


prompt1 = """
    A group of little girls sat under a loquat tree eating loquats,
    a basket full of them, close-up, straw hats, curly hair, big eyes, smiles,
    laughter, kittens catching butterflies, the tree covered with loquats
    """


if __name__ == '__main__':

    h = HealingIllu()
    h.text_to_image(prompt1)

