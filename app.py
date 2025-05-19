%%writefile app.py
import streamlit as st
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import io

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None
    )
    return pipe.to("cuda")

pipe = load_pipeline()

st.title("🎨 Stable Diffusion XL - Image Generator")
prompt = st.text_input("Enter your prompt:", "A fantasy landscape with castles and dragons")

if st.button("Generate"):
    with st.spinner("Generating..."):
        with torch.autocast("cuda"):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

        st.image(image, caption="Generated Image", use_column_width=True)

        # Optionally provide a download button
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        st.download_button("Download Image", data=img_bytes.getvalue(), file_name="generated.png", mime="image/png")
