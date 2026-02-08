import streamlit as st
import torch
from PIL import Image
from geoclip import GeoCLIP
import folium
from streamlit_folium import st_folium

# 1. Page Configuration
st.set_page_config(page_title="AI GeoGuessr Solver", layout="wide")

@st.cache_resource
def load_model():
    # Load the GeoCLIP model (will download weights on first run)
    return GeoCLIP()

st.title("🌍 AI GeoGuessr Location Detector")
st.markdown("Analyzing global coordinates using Vision Transformers.")

# 2. Loading State
try:
    with st.spinner("Initializing AI Brain (Loading weights)..."):
        model = load_model()
except Exception as e:
    st.error(f"Failed to load the model. Error: {e}")

# 3. Sidebar for Input
with st.sidebar:
    st.header("Step 1: Upload Image")
    uploaded_file = st.file_uploader("Upload a Street View screenshot", type=['jpg', 'jpeg', 'png'])
    st.info("The AI looks for: utility poles, tree species, road lines, and sun position.")

# 4. Main App Logic
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    # Load and convert image
    input_image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("Target Image")
        # FIX: Changed use_container_width to use_column_width for stability
        st.image(input_image, caption="Uploaded Scene", use_column_width=True)
        
    with col2:
        st.subheader("AI Prediction")
        with st.spinner("Analyzing world clues..."):
            # GeoCLIP predict returns (top_pred_gps, top_pred_prob)
            top_k_preds, top_k_probs = model.predict(input_image, top_k=3)
            
            # Extract the first prediction
            # top_k_preds is typically a list of [lat, lon]
            best_guess = top_k_preds[0]
            lat, lon = float(best_guess[0]), float(best_guess[1])
            
            st.success(f"Best Guess: {lat:.4f}, {lon:.4f}")
            
            # Create the map
            m = folium.Map(location=[lat, lon], zoom_start=4)
            folium.Marker(
                [lat, lon], 
                popup="AI Prediction", 
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # FIX: st_folium width set to None to auto-fill the column
            st_folium(m, width=None, height=450)
else:
    st.info("Waiting for an image upload in the sidebar.")
    # Show a neutral world map
    m_empty = folium.Map(location=[20, 0], zoom_start=2)
    st_folium(m_empty, width=None, height=450)
