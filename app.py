
import streamlit as st
import numpy as np
import pickle
from tensorflow import keras
import keras.src as keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from PIL import Image
import requests
from io import BytesIO

# Preprocess the uploaded image
def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image)
    image = image.resize((224, 224))  # Resize the image to match VGG16 input size
    image = np.array(image)
    image = preprocess_input(image)
    return image

# Preprocess the image from URL
def preprocess_image_url(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image = image.resize((224, 224))  # Resize the image to match VGG16 input size
    image = np.array(image)
    image = preprocess_input(image)
    return image

# Load MobileNetV2 model
model_1 = load_model("best_mode_vgg_40.h5", compile=False)

# Load the tokenizer
with open("tokenizer40_new.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate captions
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating the next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text

def main():
    st.title("Image Caption Generator 📷 ➡️ 📝")

    # Choose an input option: Upload or URL
    input_option = st.radio("Select an input option:", ("Upload Image", "Image URL"))

    if input_option == "Upload Image":
        # Upload an image
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Generate caption button
            if st.button("Generate Caption"):
                # Preprocess the uploaded image
                new_image = preprocess_image(uploaded_image)

                # Generate features for the new image using the pre-trained VGG16 model
                vgg_model = VGG16()
                vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
                new_image_features = vgg_model.predict(np.array([new_image]), verbose=0)

                # Predict caption for the new image
                generated_caption = predict_caption(model_1, new_image_features, tokenizer, max_length=35)
                generated_caption = generated_caption.replace('startseq', '').replace('endseq', '').strip()
                generated_caption = generated_caption.capitalize()

                # Display the generated caption
                st.markdown('#### Predicted Captions:')
                st.markdown(f"<p style='font-size:25px'><i>{generated_caption}</i>.</p>",
                            unsafe_allow_html=True)

    elif input_option == "Image URL":
        # Input image URL
        image_url = st.text_input("Enter the image URL:")

        if image_url:
            # Display the image
            image = Image.open(BytesIO(requests.get(image_url).content))
            st.image(image, caption="Image", use_column_width=True)

            # Generate caption button
            if st.button("Generate Caption"):
                # Preprocess the image from URL
                new_image = preprocess_image_url(image_url)

                # Generate features for the new image using the pre-trained VGG16 model
                vgg_model = VGG16()
                vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
                new_image_features = vgg_model.predict(np.array([new_image]), verbose=0)

                # Generate caption for the new image
                generated_caption = predict_caption(model_1, new_image_features, tokenizer, max_length=35)
                generated_caption = generated_caption.replace('startseq', '').replace('endseq', '').strip()
                generated_caption = generated_caption.capitalize()

                # Display the generated caption
                st.markdown('#### Predicted Captions:')
                st.markdown(f"<p style='font-size:25px'><i>{generated_caption}</i>.</p>",
                            unsafe_allow_html=True)

if __name__ == "__main__":
    main()
