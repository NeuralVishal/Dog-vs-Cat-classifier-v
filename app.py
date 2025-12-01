import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = load_model("cat_dog_vgg16_clean.h5")

st.title("🐶🐱 Dog vs Cat Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocessing (same as training)
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    img_array = img_array / 255.0  # normalization

    # Predict
    prediction = model.predict(img_array)

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    if prediction[0][0] > 0.5:
        st.success("It's a Dog 🐶")
    else:
        st.success("It's a Cat 🐱")
