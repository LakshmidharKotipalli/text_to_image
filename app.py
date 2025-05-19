import streamlit as st
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import io
import os

# Load pipeline only once and cache
@st.cache_resource
def load_pipeline():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        safety_checker=None,
        token=os.getenv("HF_TOKEN")  # Read token securely from Streamlit secrets
    )

    return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Load model
pipe = load_pipeline()

# Streamlit UI
st.set_page_config(page_title="SDXL Image Generator", layout="centered")
st.title("🎨 Stable Diffusion XL - Text-to-Image Generator")
st.markdown("Generate high-quality AI images using Stability AI's SDXL 1.0 model.")

prompt = st.text_input("Enter a prompt to generate an image:", "A futuristic city with flying cars at sunset")

num_steps = st.slider("Inference steps", min_value=10, max_value=100, value=40)
guidance = st.slider("Guidance scale (higher = more prompt influence)", 1.0, 20.0, 7.5)
image_width = st.selectbox("Image width", [512, 768, 1024], index=1)
image_height = st.selectbox("Image height", [512, 768, 1024], index=1)

if st.button("Generate Image"):
    st.info("Generating image, please wait...")

    if torch.cuda.is_available():
        with torch.autocast("cuda"):
            result = pipe(prompt, height=image_height, width=image_width,
                          num_inference_steps=num_steps, guidance_scale=guidance)
    else:
        result = pipe(prompt, height=image_height, width=image_width,
                      num_inference_steps=num_steps, guidance_scale=guidance)

    image = result.images[0]
    st.image(image, caption="🖼️ Generated Image", use_column_width=True)

    # Download button
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    st.download_button(
        label="Download Image",
        data=img_bytes.getvalue(),
        file_name="generated_image.png",
        mime="image/png"
    )
