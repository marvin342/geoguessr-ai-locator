import streamlit as st
import torch
from PIL import Image
from geoclip import GeoCLIP
import folium
from streamlit_folium import folium_static  # Changed this line

# 1. Page Configuration
st.set_page_config(page_title="AI GeoGuessr Solver", layout="wide")

@st.cache_resource
def load_model():
    # Load the GeoCLIP model
    return GeoCLIP()

st.title("🌍 AI GeoGuessr Location Detector")

# 2. Loading State
try:
    with st.spinner("Loading AI Brain..."):
        model = load_model()
except Exception as e:
    st.error(f"Model Error: {e}")

# 3. Sidebar
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Screenshot from Google Street View", type=['jpg', 'jpeg', 'png'])

# 4. Main App Logic
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    input_image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("Your Image")
        st.image(input_image, use_column_width=True)
        
    with col2:
        st.subheader("AI Prediction")
        with st.spinner("Analyzing coordinates..."):
            # Get predictions
            top_k_preds, top_k_probs = model.predict(input_image, top_k=1)
            
            # Extract Latitude and Longitude
            best_guess = top_k_preds[0]
            lat, lon = float(best_guess[0]), float(best_guess[1])
            
            st.success(f"Best Guess: {lat:.4f}, {lon:.4f}")
            
            # Create the map
            m = folium.Map(location=[lat, lon], zoom_start=4)
            folium.Marker([lat, lon], popup="AI Guess").add_to(m)
            
            # Use folium_static for stability (Fixes MarshallComponentException)
            folium_static(m, width=600, height=450)
else:
    st.info("Upload an image in the sidebar to begin.")
    # Show a static empty map
    m_empty = folium.Map(location=[20, 0], zoom_start=2)
    folium_static(m_empty, width=600, height=450)
