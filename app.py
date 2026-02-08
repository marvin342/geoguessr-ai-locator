import streamlit as st
import torch
from PIL import Image
from geoclip import GeoCLIP
import folium
from streamlit_folium import folium_static
import tempfile
import os
import gc

# 1. Page Config - Keep it simple
st.set_page_config(page_title="AI Locator", layout="centered")

@st.cache_resource
def load_model():
    # Force everything to CPU and set to evaluation mode
    model = GeoCLIP()
    model.to("cpu")
    model.eval()
    return model

st.title("🌍 GeoGuessr AI (CPU Mode)")

# Clear memory immediately on load
gc.collect()

# Load the AI
with st.spinner("Waking up the AI..."):
    try:
        model = load_model()
    except Exception as e:
        st.error("Server is too busy. Click 'Reboot' in settings.")

# 2. Sidebar Upload
uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'png'])

if uploaded_file:
    # Open image and resize it to be smaller (saves RAM)
    img = Image.open(uploaded_file).convert("RGB")
    img.thumbnail((500, 500)) # Shrink the image for the CPU
    st.image(img, caption="Analyzing this view...")
    
    with st.spinner("Thinking..."):
        # Save a small temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            img.save(tmp.name, quality=40)
            tmp_path = tmp.name
        
        try:
            # Inference without gradients to save memory
            with torch.no_grad():
                preds, probs = model.predict(tmp_path, top_k=1)
            
            lat, lon = float(preds[0][0]), float(preds[0][1])
            st.success(f"AI Guess: {lat:.2f}, {lon:.2f}")
            
            # Simple Map
            m = folium.Map(location=[lat, lon], zoom_start=4)
            folium.Marker([lat, lon]).add_to(m)
            folium_static(m)
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            # FORCE memory cleanup
            del img
            gc.collect()
else:
    st.info("Upload a street view picture to start.")
