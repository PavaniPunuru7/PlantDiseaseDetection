import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import requests
from PIL import Image
from io import BytesIO

# ----------- Constants -----------
MODEL_PATH = "CNN_plantdiseases_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ----------- Load model -----------
if not os.path.exists(MODEL_PATH):
    st.sidebar.error(f"❌ Model file not found: {MODEL_PATH}")
    model = None
else:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")
        model = None

# ----------- Prediction function -----------
def model_predict(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = img.reshape(1, *IMG_SIZE, 3)
        prediction = np.argmax(model.predict(img), axis=-1)[0]
        return prediction
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        return None

# ----------- Sidebar and Page Navigation -----------
st.sidebar.title("🌱 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# ----------- Show banner image -----------
try:
    image_url = "https://res.cloudinary.com/dg6y4ysiq/image/upload/v1753350327/plant_bqzzvs.jpg"
    response = requests.get(image_url)
    banner_image = Image.open(BytesIO(response.content))
    st.image(banner_image, caption="Healthy Plant", use_container_width=True)
except:
    st.warning("⚠️ Unable to load banner image from URL.")

# ----------- Home Page -----------
if app_mode == "HOME":
    st.markdown(
        "<h1 style='text-align:center;'>Plant Disease Detection System for Sustainable Agriculture</h1>",
        unsafe_allow_html=True,
    )

# ----------- Prediction Page -----------
elif app_mode == "DISEASE RECOGNITION":
    st.header("🌿 Upload Plant Leaf Image to Diagnose")
    test_image = st.file_uploader("📷 Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        save_path = os.path.join(os.getcwd(), test_image.name)

        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

        st.image(test_image, use_container_width=True, caption="Uploaded Image")

        if st.button("Predict"):
            if model is None:
                st.error("❌ Model is not loaded. Please check model file.")
            else:
                st.snow()
                st.write("🔍 Analyzing image...")
                result_index = model_predict(save_path)

                if result_index is not None and result_index < len(CLASS_NAMES):
                    st.success(f"✅ Prediction: **{CLASS_NAMES[result_index]}**")
                else:
                    st.error("⚠️ Prediction failed or index out of range.")
