import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, UnidentifiedImageError

# Load the trained model
model = load_model("plant_disease_model.h5")

# Class labels and pesticides
DISEASE_INFO = {
    0: {"name": "Tomato Bacterial Spot", "pesticide": "Copper-based fungicides"},
    1: {"name": "Potato Early Blight", "pesticide": "Mancozeb or Chlorothalonil"},
    2: {"name": "Corn Common Rust", "pesticide": "Azoxystrobin or Propiconazole"},
}

# Preprocess the image to match the model's input requirements
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((256, 256))  # Resize to the model's expected size
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit App
st.title("🌱 Plant Disease Det. & Pest. Rec")

# Initialize session state
if "predicted_disease" not in st.session_state:
    st.session_state["predicted_disease"] = None
if "recommended_pesticide" not in st.session_state:
    st.session_state["recommended_pesticide"] = None

# Upload Image Section
plant_image = st.file_uploader(
    "Upload an image (supported: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
)

if plant_image:
    try:
        image = Image.open(plant_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict Disease"):
            # Preprocess and predict
            preprocessed_image = preprocess_image(image)
            predictions = model.predict(preprocessed_image)
            predicted_class = np.argmax(predictions)
            
            # Debugging: Check the predictions
            st.write(f"Predictions: {predictions}")
            st.write(f"Predicted Class: {predicted_class}")
            
            # Update session state with predictions
            disease_info = DISEASE_INFO.get(
                predicted_class, 
                {"name": "Unknown Disease", "pesticide": "No recommendation available"}
            )
            st.session_state["predicted_disease"] = disease_info["name"]
            st.session_state["recommended_pesticide"] = disease_info["pesticide"]

            # Display the results
            st.success(f"**Disease Detected:** {disease_info['name']}")
            st.info(f"**Recommended Pesticide:** {disease_info['pesticide']}")
    except UnidentifiedImageError:
        st.error("Invalid image file. Please upload a valid image.")

# Summary Section
st.header("Summary")
if st.session_state["predicted_disease"]:
    st.write(f"**Disease:** {st.session_state['predicted_disease']}")
    st.write(f"**Pesticide:** {st.session_state['recommended_pesticide']}")
else:
    st.warning("No predictions made yet. Upload an image and predict first.")

# Disclaimer Section
st.markdown(
    """
    ---
    **Disclaimer:** This app provides general recommendations. 
    For detailed guidance, consult a local agricultural expert.
    """
)
