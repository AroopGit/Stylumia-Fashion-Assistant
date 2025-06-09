import streamlit as st
import requests
from PIL import Image
import io
import json
import time

# Update API URL to the deployed backend
API_URL = "https://stylumia-fashion-assistant.onrender.com"

# Set page config
st.set_page_config(
    page_title="Stylumia Outfit Recommender",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for wishlist if it doesn't exist
if 'wishlist' not in st.session_state:
    st.session_state['wishlist'] = []

# Add retry mechanism for health check
def check_backend_health(max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            health = requests.get(f"{API_URL}/health", timeout=10)
            if health.status_code == 200:
                health_data = health.json()
                if health_data["status"] == "healthy":
                    return True
            if attempt < max_retries - 1:
                time.sleep(delay)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            continue
    return False

# Custom CSS for mobile responsiveness and style
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .outfit-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
    }
    .item-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1rem;
        background-color: white;
        border-radius: 8px;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        min-width: 180px;
        max-width: 220px;
        flex: 1 1 180px;
    }
    .price-tag {
        color: #FF4B4B;
        font-weight: bold;
        font-size: 1.2rem;
    }
    @media (max-width: 900px) {
        .outfit-container {
            flex-direction: column;
            align-items: center;
        }
        .item-container {
            min-width: 90vw;
            max-width: 95vw;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Wishlist ---
st.sidebar.title("❤️ My Wishlist")
st.sidebar.markdown("---")

# Display wishlist items in sidebar
if st.session_state['wishlist']:
    for idx, item in enumerate(st.session_state['wishlist']):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.sidebar.write(f"• {item}")
        with col2:
            if st.sidebar.button("🗑️", key=f"remove_{idx}"):
                st.session_state['wishlist'].pop(idx)
                st.rerun()
else:
    st.sidebar.info("Your wishlist is empty. Add items to get started!")

# --- Main UI ---
st.title("Stylumio Outfit Recommender")
st.markdown("""
    Discover your perfect outfit! Upload an image or select a style to get personalized fashion recommendations.
    Our AI-powered system will help you find the perfect combination of clothing items.
""")

# Health check with retry mechanism
if not check_backend_health():
    st.error("""
        The backend service is currently starting up or experiencing issues. 
        Please wait a moment and refresh the page. 
        This is normal for the first request after a period of inactivity.
    """)
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Style-Based Recommendations")
    style = st.selectbox(
        "Select Your Style",
        ["casual", "formal", "business", "evening", "sporty"],
        format_func=lambda x: x.capitalize()
    )
    
    if st.button("Get Style Recommendations"):
        with st.spinner("Finding the perfect outfit for your style..."):
            try:
                resp = requests.get(f"{API_URL}/api/outfit-recommendations/{style}", timeout=30)
                if resp.status_code == 200:
                    outfits = resp.json()
                    for idx, outfit in enumerate(outfits):
                        st.markdown(f"### {outfit['style'].capitalize()} Outfit")
                        item_cols = st.columns(len(outfit["items"]))
                        for item_idx, (item, col) in enumerate(zip(outfit["items"], item_cols)):
                            with col:
                                st.image(item['image_url'], width=200)
                                st.markdown(f"**{item['product_name']}**")
                                st.write(f"by {item['brand']}")
                                st.markdown(f"<span style='color:#FF4B4B;font-weight:bold;'>${item['price']:.2f}</span>", unsafe_allow_html=True)
                                if st.button("💖 Add to Wishlist", key=f"wishlist_{item['product_name']}_{idx}_{item_idx}"):
                                    name = item.get('product_name', 'Unknown Item')
                                    if name not in st.session_state['wishlist']:
                                        st.session_state['wishlist'].append(name)
                                        st.rerun()
                else:
                    st.error(f"Error: {resp.text}")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The backend might be starting up. Please try again in a moment.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the backend: {str(e)}")

with col2:
    st.subheader("Image-Based Recommendations")
    uploaded_file = st.file_uploader(
        "Upload an image for personalized recommendations",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a clothing item to get matching outfit recommendations"
    )
    
    if uploaded_file and st.button("Get Image-Based Recommendations"):
        with st.spinner("Analyzing your image and finding perfect matches..."):
            try:
                resp = requests.post(
                    f"{API_URL}/api/outfit-recommendations/upload",
                    files={"file": uploaded_file},
                    data={"style": style},
                    timeout=60  # Longer timeout for image processing
                )
                if resp.status_code == 200:
                    outfits = resp.json()
                    if outfits:
                        st.markdown(f"### {outfits[0]['style'].capitalize()} Outfit")
                        all_items = []
                        for outfit in outfits:
                            all_items.extend(outfit["items"])
                        for i in range(0, len(all_items), 3):
                            row_items = all_items[i:i+3]
                            cols = st.columns(3)
                            for col, item in zip(cols, row_items):
                                with col:
                                    st.image(item['image_url'], width=200)
                                    st.markdown(f"**{item['product_name']}**")
                                    st.write(f"by {item['brand']}")
                                    st.markdown(f"<span style='color:#FF4B4B;font-weight:bold;'>${item['price']:.2f}</span>", unsafe_allow_html=True)
                                    if st.button("💖 Add to Wishlist", key=f"wishlist_{item['product_name']}_img_{i}"):
                                        name = item.get('product_name', 'Unknown Item')
                                        if name not in st.session_state['wishlist']:
                                            st.session_state['wishlist'].append(name)
                                            st.rerun()
                else:
                    st.error(f"Error: {resp.text}")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The backend might be starting up. Please try again in a moment.")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to the backend: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Made by Aroop :) </p>
    </div>
""", unsafe_allow_html=True)