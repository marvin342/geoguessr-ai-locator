import streamlit as st
from geoclip import GeoCLIP
import folium
from streamlit_folium import st_folium
import torch
from PIL import Image
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="GeoGuessr AI Locator", layout="centered")

# --- POWERFUL CACHING (The "Fix") ---
# This tells the server: "Load this model ONCE and keep it in the background."
@st.cache_resource
def load_model():
    # Force CPU mode to save memory
    device = torch.device('cpu')
    model = GeoCLIP()
    model.to(device)
    model.eval()
    return model

# --- LOAD DATA ---
st.title("🌍 GeoGuessr AI Locator")
st.write("Upload a GeoGuessr screenshot, and the AI will guess the location!")

with st.spinner("Loading AI Model... (This takes a minute on first run)"):
    model = load_model()

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Choose a screenshot...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Screenshot', use_container_width=True)
    
    # Save temporarily to process
    temp_path = "temp_coords.jpg"
    image.save(temp_path)
    
    if st.button("📍 Predict Location"):
        with st.spinner("Analyzing landscape and road markings..."):
            try:
                # Run the prediction
                with torch.no_grad():
                    preds, probs = model.predict(temp_path, top_k=1)
                
                lat, lon = float(preds[0][0]), float(preds[0][1])
                
                st.success(f"AI Guess: {lat:.2f}, {lon:.2f}")
                
                # --- MAP DISPLAY ---
                m = folium.Map(location=[lat, lon], zoom_start=4)
                folium.Marker([lat, lon], popup="AI Guess").add_to(m)
                st_folium(m, width=700, height=450)
                
            except Exception as e:
                st.error(f"Something went wrong: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

st.info("Note: This AI uses visual cues like foliage, sun position, and infrastructure.")
