import sys
import os
import dotenv
from google import genai
from google.genai import types
from PIL import Image
import base64
from io import BytesIO

# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/imagen-api?hl=ja

def main():
    client = genai.Client(
        vertexai=True,
        project=os.getenv("PROJECT_ID"),
        location=os.getenv("LOCATION"),
    )

    generation_model = "imagen-3.0-generate-002"
    prompt = "A fantasy landscape, trending on artstation"
    response = client.models.generate_images(
        model=generation_model,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="3:4",
            safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
            person_generation="DONT_ALLOW",
        )
    )

# https://googleapis.github.io/python-genai/genai.html#genai.types.Image
    for i, image in enumerate(response.images):
        image.save(location=f"generated_image_{i}.png")



if __name__ == "__main__":
    main()