import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import requests
from PIL import Image
from io import BytesIO

# ----------- Model download logic (Dropbox) -----------
MODEL_PATH = "CNN_plantdiseases_model.keras"
MODEL_URL = "https://www.dropbox.com/scl/fi/m852g1gh7ghpx97k79l9i/CNN_plantdiseases_model.keras?rlkey=pg5c13l9970vt1qy9abqpdtqg&st=gt6l89zi&dl=1"

def download_model():
    print("📦 Downloading model from Dropbox...")
    response = requests.get(MODEL_URL, stream=True)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(1024 * 1024):  # 1MB chunks
                f.write(chunk)
        print("✅ Model downloaded successfully.")
    else:
        raise Exception(f"❌ Failed to download model. Status code: {response.status_code}")

# Download if not already present
if not os.path.exists(MODEL_PATH):
    download_model()

# Load and process the image
def model_predict(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    img = cv2.imread(image_path)
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, H, W, C)
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Sidebar
st.sidebar.title("🌱 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display image from URL
image_url = "https://res.cloudinary.com/dg6y4ysiq/image/upload/v1753350327/plant_bqzzvs.jpg"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))
st.image(img, caption="Healthy Plant", use_container_width=True)

# Main Page
if app_mode == "HOME":
    st.markdown(
        "<h1 style='text-align:center;'>Plant Disease Detection System for Sustainable Agriculture</h1>",
        unsafe_allow_html=True,
    )

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("🌿 Upload Plant Leaf Image to Diagnose")
    test_image = st.file_uploader("📷 Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        save_path = os.path.join(os.getcwd(), test_image.name)

        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

        if st.button("Show Image"):
            st.image(test_image, use_container_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("🔍 Analyzing...")
            result_index = model_predict(save_path)
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
                'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]
            st.success(f"✅ Prediction: **{class_name[result_index]}**")
