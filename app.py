import streamlit as st
import torch
from PIL import Image
from geoclip import GeoCLIP
import folium
from streamlit_folium import st_folium

# Page Configuration
st.set_page_config(page_title="AI GeoGuessr Solver", layout="wide")

@st.cache_resource
def load_model():
    # Downloads the weights (~700MB) from the official repo
    return GeoCLIP()

st.title("🌍 AI GeoGuessr Location Detector")
st.markdown("Upload a Street View screenshot and the AI will guess the location based on vegetation, road lines, and sun position.")

# Load the AI
with st.spinner("Loading AI Brain..."):
    model = load_model()

# UI Layout
col1, col2 = st.columns([1, 1])

with st.sidebar:
    st.header("Step 1: Upload")
    uploaded_file = st.file_uploader("Choose a Street View image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(input_image, caption="Uploaded Image", use_container_width=True)
        
    with col2:
        with st.spinner("Analyzing world clues..."):
            # Get predictions
            top_k_preds, top_k_probs = model.predict(input_image, top_k=3)
            
            # Get the best guess (first item in the list)
            best_guess = top_k_preds[0]
            lat, lon = best_guess[0], best_guess[1]
            
            st.success(f"Best Guess: {lat:.4f}, {lon:.4f}")
            
            # Display the Map
            m = folium.Map(location=[lat, lon], zoom_start=4)
            folium.Marker([lat, lon], popup="AI Prediction #1", icon=folium.Icon(color='red')).add_to(m)
            
            # Show map in Streamlit
            st_folium(m, width=700, height=450)
else:
    st.info("Please upload an image in the sidebar to begin.")
