import streamlit as st
import torch
from PIL import Image
from geoclip import GeoCLIP
import folium
from streamlit_folium import folium_static
import tempfile
import os

# 1. Page Config
st.set_page_config(page_title="GeoGuessr AI", layout="wide")

# 2. Model Loader (Cached)
@st.cache_resource
def load_model():
    # Force CPU mode for stability on Streamlit's shared servers
    model = GeoCLIP()
    model.to("cpu") 
    return model

st.title("🌍 Professional GeoGuessr AI")
st.write("Detecting locations using world-scale Vision Transformers.")

# Load Model
with st.spinner("Downloading AI weights... (Wait for balloons 🎈)"):
    model = load_model()

# 3. Sidebar for Upload
with st.sidebar:
    st.header("Upload Screenshot")
    uploaded_file = st.file_uploader("Street View image", type=['jpg', 'jpeg', 'png'])
    st.info("The AI analyzes architecture, vegetation, and road markings.")

# 4. Processing Logic
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    # Open and show the image
    img = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("Your Image")
        st.image(img, use_column_width=True)
        
    with col2:
        st.subheader("AI Guess")
        with st.spinner("Analyzing pixels for geographic clues..."):
            # SAVE TO TEMP FILE (Essential for GeoCLIP API)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                img.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                # FIX: Use torch.no_grad() to save memory and prevent 'F.linear' weight update errors
                with torch.no_grad():
                    top_k_preds, top_k_probs = model.predict(tmp_path, top_k=1)
                
                # FIX: Explicitly cast to float to prevent TypeError in Folium/JSON Marshalling
                lat = float(top_k_preds[0][0])
                lon = float(top_k_preds[0][1])
                
                st.success(f"Coordinate Match: {lat:.4f}, {lon:.4f}")
                
                # Render the map
                m = folium.Map(location=[lat, lon], zoom_start=5)
                folium.Marker([lat, lon], popup="AI Guess", icon=folium.Icon(color='red')).add_to(m)
                folium_static(m, width=650, height=450)
                
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
                st.warning("If this persists, go to Settings -> Advanced -> Python Version and select 3.11")
                
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
else:
    st.info("Please upload an image to begin.")
    m_default = folium.Map(location=[20, 0], zoom_start=2)
    folium_static(m_default, width=650, height=450)
