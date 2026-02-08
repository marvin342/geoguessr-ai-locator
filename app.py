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

# 2. Model Loader with "Unpacker" Logic
@st.cache_resource
def load_model():
    model = GeoCLIP()
    model.to("cpu")
    model.eval() # Set to evaluation mode
    return model

st.title("🌍 Professional GeoGuessr AI")

with st.spinner("Loading AI Brain..."):
    model = load_model()

# 3. Sidebar
with st.sidebar:
    st.header("Upload Screenshot")
    uploaded_file = st.file_uploader("Street View image", type=['jpg', 'jpeg', 'png'])

# 4. Processing Logic
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    img = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("Your Image")
        st.image(img, use_column_width=True)
        
    with col2:
        st.subheader("AI Guess")
        with st.spinner("Analyzing geographic signatures..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                img.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                # IMPORTANT: Use torch.no_grad to prevent the 'BaseModelOutput' error
                with torch.no_grad():
                    # We manually trigger a simple inference to clear the buffer
                    top_k_preds, top_k_probs = model.predict(tmp_path, top_k=1)
                
                lat = float(top_k_preds[0][0])
                lon = float(top_k_preds[0][1])
                
                st.success(f"Best Match: {lat:.4f}, {lon:.4f}")
                
                m = folium.Map(location=[lat, lon], zoom_start=5)
                folium.Marker([lat, lon], popup="AI Guess").add_to(m)
                folium_static(m, width=600, height=450)
                
            except Exception as e:
                # Special handling for the error in your screenshot
                if "BaseModelOutput" in str(e) or "linear" in str(e):
                    st.error("⚠️ Compatibility Error Detected")
                    st.warning("Please ensure you have changed the Python Version to 3.11 in the 'Advanced Settings' of Streamlit Cloud.")
                else:
                    st.error(f"Prediction Error: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
else:
    st.info("Upload an image to start.")
    m_default = folium.Map(location=[20, 0], zoom_start=2)
    folium_static(m_default, width=600, height=450)
