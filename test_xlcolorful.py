from sdxl import SDXLModel, SDModelWrap


class XLColorful(SDModelWrap):

    # Base Model: SDXL

    _inference_config = {
        "prompt_prefix": "Best quality, masterpiece,8k, ultra-detailed",
        "negative_prompt_prefix": """ng_deepnegative_v1_75t,(badhandv4:1.2),EasyNegative,(worst quality:2)""",
        "width": 1040,
        "height": 1664,
        "num_inference_steps": 30,
        "guidance_scale": 7,
        "sampler": "EulerA",
    }

    def __init__(self):
        super().__init__()
        self._model_name = "XL-Colorful_Anss_V1"
        self._model = SDXLModel(self._model_name, self._inference_config)


prompt1 = """
    girl with vibrant Harajuku fashion, wearing intricately detailed transparent color PVC and vinyl clothing with prismatic, 
    holographic effects, and subtle chromatic aberration, gazing directly at the viewer with an captivating expression
    """


if __name__ == '__main__':

    x = XLColorful()
    x.text_to_image(prompt1)

