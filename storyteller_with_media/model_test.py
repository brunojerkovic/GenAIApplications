from prompt_generation import OpenAILLM
from dotenv import load_dotenv
load_dotenv()

model = OpenAILLM(model_name="gpt-3.5-turbo-0125", temperature=1.)
#a, b = model.create_story("Story about a man that became a laptop.", 400, 5)
#print()
"""
response = model.client.images.generate(
    model="dall-e-3",
    prompt="Generate image of a cat",
    size="1024x1024",
    quality="standard",
    n=1,
)
path = model.get_new_image(response.data[0].url, None)

from PIL import Image
import numpy as np
image = Image.open(path)
print()
"""
path = "generated_image_DATt.png"
path = "new_dog.png"

response = model.client.images.edit(
    image=open(path, "rb"),
    mask=open(path, "rb"),
    prompt="Generate an image of dog.",
    model="dall-e-2",
    size="1024x1024",
    n=1
)

# Download the new image
import requests
image_response = requests.get(response.data[0].url)
if image_response.status_code == 200:
    filename = f"generated_image_{response.data[0].url[-15:-10]}.png".replace("\\", "").replace("/", "")
    with open(filename, "wb") as f:
        f.write(image_response.content)
else:
    filename = ""


print("HI")
