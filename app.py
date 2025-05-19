import os
from diffusers import StableDiffusionXLPipeline
import streamlit as st
from PIL import Image
import torch
import io

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        safety_checker=None,
        token=os.getenv("HF_TOKEN")
    )
    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

pipe = load_pipeline()

st.title("🎨 Stable Diffusion XL - Image Generator")
prompt = st.text_input("Enter your prompt:", "A serene mountain lake during sunrise")

if st.button("Generate"):
    with st.spinner("Generating..."):
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

        st.image(image)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        st.download_button("Download Image", data=img_bytes.getvalue(), file_name="generated.png", mime="image/png")
