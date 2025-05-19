import streamlit as st
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import io
import os

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="SDXL Image Generator", layout="centered")

@st.cache_resource
def load_pipeline():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        safety_checker=None,
        token=os.getenv("HF_TOKEN")
    )

    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Load the pipeline
pipe = load_pipeline()

# Streamlit UI
st.title("🎨 Stable Diffusion XL - Text-to-Image Generator")
st.markdown("Generate stunning images using Stability AI’s SDXL 1.0 model.")

# Input fields
prompt = st.text_input("Enter a prompt:", "A futuristic city with flying cars at sunset")
num_steps = st.slider("Inference steps", min_value=10, max_value=100, value=40)
guidance = st.slider("Guidance scale", 1.0, 20.0, 7.5)
width = st.selectbox("Image width", [512, 768, 1024], index=1)
height = st.selectbox("Image height", [512, 768, 1024], index=1)

# Generate button
if st.button("Generate Image"):
    st.info("⏳ Generating image...")

    if torch.cuda.is_available():
        with torch.autocast("cuda"):
            result = pipe(prompt, height=height, width=width,
                          num_inference_steps=num_steps, guidance_scale=guidance)
    else:
        result = pipe(prompt, height=height, width=width,
                      num_inference_steps=num_steps, guidance_scale=guidance)

    image = result.images[0]
    st.image(image, caption="🖼️ Generated Image", use_column_width=True)

    # Download button
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    st.download_button("Download Image", data=img_bytes.getvalue(), file_name="generated_image.png", mime="image/png")
