from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionPipeline
from transformers import pipeline
import torch
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def generate_image(prompt: str):
    image = pipe(prompt).images[0]
    image_path = f"generated_images/{hash(prompt)}.png"
    image.save(image_path)
    return image_path

generator = pipeline("text-generation", model="facebook/opt-1.3b")


def generate_text(prompt: str):
    result = generator(prompt, max_length=100, num_return_sequences=3)
    return [text["generated_text"] for text in result]


@app.get("/")
def read_root():
    return {"message": "API работает!"}


@app.post("/generate_post/")
def generate_post(prompt: str):
    text = generate_text(prompt)
    image_url = generate_image(prompt)
    return {"text": text, "image": image_url}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
