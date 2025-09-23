import sys
import os
import dotenv
from google import genai
from google.genai import types
from PIL import Image
import base64
from io import BytesIO

# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/imagen-api?hl=ja

def exp01(model, client, prompt=None):
    #https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateImagesConfig
    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
        )
    )
    # https://googleapis.github.io/python-genai/genai.html#genai.types.Image
    for i, image in enumerate(response.images):
        image.save(location=f"generated_image_{i}.png")

def exp02(model, client, prompt=None):

    # enhance_prompt=Trueを指定すると、プロンプトのリライトが行われる。
    prompt = "A fantasy landscape, trending on artstation"
    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            enhance_prompt=True
        )
    )
    #print("Enhanced Prompt:", response.enhanced_prompt)
    # https://googleapis.github.io/python-genai/genai.html#genai.types.Image
    for i, image in enumerate(response.images):
        image.save(location=f"generated_image_ehnaced{i}.png")

def exp03(model, client, prompt=None):
    # Photorealism and prompt understanding
    # Photorealism フォトリアリスティックな画像生成
    # Prompt 追従性の向上
    # * person_generation: DONT_ALLOW, ALLOW_ADULT, ALLOW_ALL
    # * safety_filter_level: BLOCK_LOW_AND_ABOVE, BLOCK_MEDIUM_AND_ABOVE, BLOCK_ONLY_HIGH, BLOCK_NONE

    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="1:1",
            image_size="2K",
        )
    )

    # https://googleapis.github.io/python-genai/genai.html#genai.types.Image
    for i, image in enumerate(response.images):
        image.save(location=f"generated_image_photorealism_{i}.png")

def exp04(model, client, prompt=None):
    # text_rendering: TEXT_RENDERING_UNSPECIFIED, ENABLED, DISABLED
    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            image_size="2K",
            safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
            person_generation="DONT_ALLOW",
        )
    )

    for i, image in enumerate(response.images):
        image.save(location=f"generated_image_text_rendering_{i}.png")

def exp05(model, client, prompt=None):
    # ウォーターマークの追加
    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            image_size="2K",
            safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
            person_generation="DONT_ALLOW",
            add_watermark=True,
        )
    )
    for i, image in enumerate(response.images):
        image.save(location=f"generated_image_watermark_{i}.png")

def main():
    client = genai.Client(
        vertexai=True,
        project=os.getenv("PROJECT_ID"),
        location=os.getenv("LOCATION"),
    )

    model = "imagen-4.0-generate-001"
    model_fast = "imagen-4.0-fast-generate-001"
    model_ultra = "imagen-4.0-ultra-generate-001"

    prompt = "Couple walking along the embankment. autum sky with twilight, beautiful, high resolution, 4k, detailed, trending on artstation"

    # 基本的な画像生成
    # exp01(model, client, prompt)

    # プロンプトエンハンスメントを使った画像生成
    #exp02(model, client, prompt)

    # Photorealism and prompt understanding
    # Photorealism フォトリアリスティックな画像生成
    # Prompt 追従性の向上
    # * person_generation: DONT_ALLOW, ALLOW_ADULT, ALLOW_ALL
    # * safety_filter_level: BLOCK_LOW_AND_ABOVE, BLOCK_MEDIUM_AND_ABOVE, BLOCK_ONLY_HIGH, BLOCK_NONE

    exp03(model_ultra, client, prompt)

    # テキストレンダリング
    prompt = """
    A panel of a comic strip. A cute gray cat is talking to a bulldog. The cat appears to be slightly disgusted. The cat says in a talk bubble: "You really seem to enjoy going outside. Fascinating." The dog responds by shrugging his shoulders. Well-articulated illustration with confident lines and shading.
    """
    exp04(model_ultra, client, prompt)

    # ウォーターマークに追加
    prompt = "delicious gourmet burger, photorealistic, 8k, high resolution, detailed, trending on artstation"
    exp05(model_ultra, client, prompt)


if __name__ == "__main__":
    main()