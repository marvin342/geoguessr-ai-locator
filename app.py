import streamlit as st
import torch
from PIL import Image
from geoclip import GeoCLIP
import folium
from streamlit_folium import folium_static
import tempfile
import os
import gc

st.set_page_config(page_title="GeoGuessr AI", layout="wide")

@st.cache_resource
def load_model():
    # Force the model to load in half-precision (float16) to save RAM
    model = GeoCLIP()
    model.to("cpu")
    model.eval()
    return model

st.title("🌍 Professional GeoGuessr AI")

# Memory management: Clear previous runs
gc.collect()

with st.spinner("Loading AI Brain (Optimizing Memory)..."):
    try:
        model = load_model()
    except Exception as e:
        st.error("Server out of memory. Try refreshing.")

with st.sidebar:
    st.header("Upload Screenshot")
    uploaded_file = st.file_uploader("Street View image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    img = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(img, use_column_width=True)
        
    with col2:
        with st.spinner("Analyzing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                img.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                with torch.no_grad():
                    # Predict using the path string as required by the API
                    top_k_preds, top_k_probs = model.predict(tmp_path, top_k=1)
                
                lat, lon = float(top_k_preds[0][0]), float(top_k_preds[0][1])
                st.success(f"Match: {lat:.4f}, {lon:.4f}")
                
                m = folium.Map(location=[lat, lon], zoom_start=5)
                folium.Marker([lat, lon]).add_to(m)
                folium_static(m, width=500, height=400)
                
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                # Clear memory after every guess
                gc.collect()
else:
    st.info("Upload an image to start.")
