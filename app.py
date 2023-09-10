import streamlit as st
from PIL import Image

# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import Model
#
# from keras.preprocessing import image
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import load_img
#
# from keras.models import load_model
# from keras.applications.vgg16 import preprocess_input
# from keras.models import Model
# from keras.utils import to_categorical
# from keras.preprocessing.text import Tokenizer
#
# model = load_model('best_model_50.h5')  # Load your trained model
# tokenizer = Tokenizer()  # Load your tokenizer
# tokenizer.fit_on_texts([])  # Update tokenizer with your specific tokens and word index
# max_length = 35 # Specify the maximum length of captions
# def idx_to_word(index, tokenizer):
#     for word, idx in tokenizer.word_index.items():
#         if idx == index:
#             return word
#     return None
# def predict_caption(model, image, tokenizer, max_length):
#     in_text = 'startseq'
#     for _ in range(max_length):
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         sequence = pad_sequences([sequence], maxlen=max_length)
#         yhat = model.predict([image, sequence], verbose=0)
#         yhat = np.argmax(yhat)
#         word = idx_to_word(yhat, tokenizer)
#         if word is None:
#             break
#         in_text += ' ' + word
#         if word == 'endseq':
#             break
#     return in_text
#
#
# def main():
#     st.title("Image Captioning App")
#
#     # Load tokenizer
#     tokenizer = load_tokenizer()
#
#     # Load pre-trained model
#     model = load_model('image_caption_model.h5')
#
#     # File uploader
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#
#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = load_image(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
#
#         # Preprocess the image
#         img = preprocess_image(image)
#
#         # Generate caption
#         caption = predict_caption(model, img, tokenizer, max_length=20)
#         st.header("Generated Caption:")
#         st.write(caption)
#
# if __name__ == '__main__':
#     main()

# import streamlit as st
# import preprocessing
# import matplotlib.pyplot as plt
# import helper
# st.sidebar.title("Whatsapp chat analyzer")




# import streamlit as st
# import numpy as np
# from PIL import Image
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.models import Model
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import load_model
# from keras.preprocessing.text import Tokenizer
# import pickle
#
# def train_tokenizer(texts):
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(texts)
#     return tokenizer
#
# def load_tokenizer():
#     with open('tokenizer50.pickle', 'rb') as handle:
#         tokenizer = pickle.load(handle)
#     return tokenizer
#
# def load_image(image_file):
#     img = Image.open(image_file)
#     return img
#
# def preprocess_image(image):
#     img = image.resize((224, 224))  # Resize the image to match the input size of VGG16
#     img = np.array(img)
#     img = np.expand_dims(img, axis=0)
#     img = preprocess_input(img)
#
#     # Load VGG16 model to extract image features
#     base_model = VGG16(weights='imagenet')
#     model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
#
#     # Extract image features
#     features = model.predict(img)
#     features = np.reshape(features, (1, 4096))  # Reshape features to match the expected input shape
#
#     return features
#
# def predict_caption(model, image, tokenizer, max_length):
#     in_text = 'startseq'
#     for _ in range(max_length):
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         sequence = pad_sequences([sequence], maxlen=max_length)
#         yhat = model.predict([image, sequence], verbose=0)
#         yhat = np.argmax(yhat)
#         word = idx_to_word(yhat, tokenizer)
#         if word is None:
#             break
#         in_text += ' ' + word
#         if word == 'endseq':
#             break
#     return in_text
#
# def idx_to_word(index, tokenizer):
#     for word, idx in tokenizer.word_index.items():
#         if idx == index:
#             return word
#     return None
#
# def main():
#     st.title("Image Caption Generator App")
#
#     # Check if tokenizer exists
#     try:
#         tokenizer = load_tokenizer()
#     except FileNotFoundError:
#         st.write("Tokenizer not found. Please provide a list of texts to train the tokenizer.")
#         return
#
#     # Load pre-trained model
#     model = load_model('best_model_50.h5')
#
#     # File uploader
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#
#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = load_image(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
#
#         # Preprocess the image
#         img = preprocess_image(image)
#
#         # Generate caption
#         caption = predict_caption(model, img, tokenizer, max_length=20)
#         st.header("Generated Caption:")
#         st.write(caption)
#
# if __name__ == '__main__':
#     main()



# import streamlit as st
# import numpy as np
# from PIL import Image
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.models import Model
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import load_model
# from keras.preprocessing.text import Tokenizer
# import pickle
#
# def load_tokenizer():
#     with open('tokenizer50.pickle', 'rb') as handle:
#         tokenizer = pickle.load(handle)
#     return tokenizer
#
# def load_image(image_file):
#     img = Image.open(image_file)
#     return img
#
# def preprocess_image(image):
#     img = image.resize((224, 224))  # Resize the image to match the input size of VGG16
#     img = np.array(img)
#     img = np.expand_dims(img, axis=0)
#     img = preprocess_input(img)
#
#     # Load VGG16 model to extract image features
#     base_model = VGG16(weights='imagenet')
#     model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
#
#     # Extract image features
#     features = model.predict(img)
#     features = np.reshape(features, (1, 4096))  # Reshape features to match the expected input shape
#
#     return features
#
# def predict_caption(model, image, tokenizer, max_length):
#     in_text = 'startseq'
#     for _ in range(max_length):
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         sequence = pad_sequences([sequence], maxlen=max_length)
#         yhat = model.predict([image, sequence], verbose=0)
#         yhat = np.argmax(yhat)
#         word = idx_to_word(yhat, tokenizer)
#         if word is None:
#             break
#         in_text += ' ' + word
#         if word == 'endseq':
#             break
#     return in_text
#
# def idx_to_word(index, tokenizer):
#     for word, idx in tokenizer.word_index.items():
#         if idx == index:
#             return word
#     return None
#
# def main():
#     st.title("Image Caption Generator App")
#
#     # Check if tokenizer exists
#     try:
#         tokenizer = load_tokenizer()
#     except FileNotFoundError:
#         st.write("Tokenizer not found. Please provide a list of texts to train the tokenizer.")
#         return
#
#     # Load pre-trained model
#     model = load_model('best_model_50.h5')
#
#     # File uploader
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#
#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = load_image(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
#
#         # Preprocess the image
#         img = preprocess_image(image)
#
#         # Generate caption
#         caption = predict_caption(model, img, tokenizer, max_length=35)
#         st.header("Generated Caption:")
#         st.write(caption)
#
# if __name__ == '__main__':
#     main()
# import streamlit as st
# import numpy as np
# from PIL import Image
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.models import Model
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import load_model
# from keras.preprocessing.text import Tokenizer
# import pickle
#
# # Load tokenizer
# def load_tokenizer():
#     with open('tokenizer50.pickle', 'rb') as handle:
#         tokenizer = pickle.load(handle)
#     return tokenizer
#
# # Preprocess the image
# def preprocess_image(image):
#     img = image.resize((224, 224))  # Resize the image to match the input size of VGG16
#     img = np.array(img)
#     img = preprocess_input(img)
#     return img
#
# # Generate caption
# def generate_caption(model, image, tokenizer, max_length):
#     in_text = 'startseq'
#     for _ in range(max_length):
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         sequence = pad_sequences([sequence], maxlen=max_length)
#         yhat = model.predict([image, sequence], verbose=0)
#         yhat = np.argmax(yhat)
#         word = idx_to_word(yhat, tokenizer)
#         if word is None:
#             break
#         in_text += ' ' + word
#         if word == 'endseq':
#             break
#     return in_text
#
# # Convert index to word
# def idx_to_word(index, tokenizer):
#     for word, idx in tokenizer.word_index.items():
#         if idx == index:
#             return word
#     return None
#
# def main():
#     st.title("Image Caption Generator App")
#
#     # Check if tokenizer exists
#     try:
#         tokenizer = load_tokenizer()
#     except FileNotFoundError:
#         st.write("Tokenizer not found. Please provide a list of texts to train the tokenizer.")
#         return
#
#     # Load pre-trained model
#     model = load_model('best_model_50.h5')
#
#     # Load VGG16 model for feature extraction
#     vgg_model = VGG16(weights='imagenet', include_top=False)
#     feature_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.output)
#
#     # File uploader
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
#
#     if uploaded_file is not None:
#         # Display the uploaded image
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)
#
#         # Preprocess the image
#         processed_image = preprocess_image(image)
#
#         # Add batch dimension to the image
#         image_batch = np.expand_dims(processed_image, axis=0)
#
#         # Extract image features
#         image_features = feature_extractor.predict(image_batch)
#
#         # Reshape the image features to match the expected shape
#         reshaped_image_features = np.reshape(image_features, (1, 7*7, 512))
#
#         # Generate caption
#         caption = generate_caption(model, reshaped_image_features, tokenizer, max_length=35)
#         st.header("Generated Caption:")
#         st.write(caption)
#
# if __name__ == '__main__':
#     main()


# import streamlit as st
# import numpy as np
# import pickle
# import tensorflow as tf
# from keras.models import Model
# from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from keras.preprocessing.image import load_img, img_to_array
# from keras.preprocessing.sequence import pad_sequences
#
# # Load MobileNetV2 model
# mobilenet_model = MobileNetV2(weights="imagenet")
# mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)
#
# # Load your trained model
# model = tf.keras.models.load_model('best_mode_vgg_40.h5')
#
# # Load the tokenizer
# with open('tokenizer40.pickle', 'rb') as tokenizer_file:
#     tokenizer = pickle.load(tokenizer_file)
#
# # Set custom web page title
# st.set_page_config(page_title="Caption Generator App", page_icon="📷")
#
# # Streamlit app
# st.title("Image Caption Generator")
# st.markdown(
#     "Upload an image, and this app will generate a caption for it using a trained LSTM model."
# )
#
# # Upload image
# uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
#
# # Process uploaded image
# if uploaded_image is not None:
#     st.subheader("Uploaded Image")
#     st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
#
#     st.subheader("Generated Caption")
#     # Display loading spinner while processing
#     with st.spinner("Generating caption..."):
#         # Load image
#         image = load_img(uploaded_image, target_size=(224, 224))
#         image = img_to_array(image)
#         image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#         image = preprocess_input(image)
#
#         # Extract features using VGG16
#         image_features = mobilenet_model.predict(image, verbose=0)
#
#         # Max caption length
#         max_caption_length = 35
#
#
#         # Define function to get word from index
#         def get_word_from_index(index, tokenizer):
#             return next(
#                 (word for word, idx in tokenizer.word_index.items() if idx == index), None
#             )
#
#
#         # Generate caption using the model
#         def predict_caption(model, image_features, tokenizer, max_caption_length):
#             caption = "startseq"
#             for _ in range(max_caption_length):
#                 sequence = tokenizer.texts_to_sequences([caption])[0]
#                 sequence = pad_sequences([sequence], maxlen=max_caption_length)
#                 yhat = model.predict([image_features, sequence], verbose=0)
#                 predicted_index = np.argmax(yhat)
#                 predicted_word = get_word_from_index(predicted_index, tokenizer)
#                 caption += " " + predicted_word
#                 if predicted_word is None or predicted_word == "endseq":
#                     break
#             return caption
#
#
#         # Generate caption
#         generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)
#
#         # Remove startseq and endseq
#         generated_caption = generated_caption.replace("startseq", "").replace("endseq", "")
#
#     # Display the generated caption with custom styling
#     st.markdown(
#         f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
#         f'<p style="font-style: italic;">“{generated_caption}”</p>'
#         f'</div>',
#         unsafe_allow_html=True
#     )
#
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def preprocess_image(image_path):
#     image = Image.open(image_path)
#     image = image.resize((224, 224))  # Resize the image to match VGG16 input size
#     image = np.array(image)
#     image = preprocess_input(image)
#     return image


import numpy as np
import pandas as pd
import cv2
import os
from glob import glob
import pickle

import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16,preprocess_input
#from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing import image

from keras.utils import load_img
from keras.utils import img_to_array

#from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences

from keras.models import Model
from keras.utils import to_categorical,plot_model
from keras.layers import Input,Dense,LSTM,Embedding,Dropout,add
# Load MobileNetV2 model

#
#
# # Load your trained model
# model = tf.keras.models.load_model('best_mode_vgg_40.h5')
#
# # Load the tokenizer
# with open('tokenizer40.pickle', 'rb') as tokenizer_file:
#     tokenizer = pickle.load(tokenizer_file)
#
# # Set custom web page title
# st.set_page_config(page_title="Caption Generator App", page_icon="📷")
#
# # Streamlit app
# st.title("Image Caption Generator")
# st.markdown(
#     "Upload an image, and this app will generate a caption for it using a trained LSTM model."
# )
#
# # Upload image
# uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
#
# # Process uploaded image
# if uploaded_image is not None:
#     st.subheader("Uploaded Image")
#     st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
#
#     st.subheader("Generated Caption")
#     # Display loading spinner while processing
#     with st.spinner("Generating caption..."):
#         # Load image
#         image = load_img(uploaded_image, target_size=(224, 224))
#         image = img_to_array(image)
#         image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#         image = preprocess_input(image)
#
#         # Extract features using VGG16
#         image_features = model.predict(image, verbose=0)
#
#         # Max caption length
#         max_caption_length = 35
#
#
#         # Define function to get word from index
#         def get_word_from_index(index, tokenizer):
#             return next(
#                 (word for word, idx in tokenizer.word_index.items() if idx == index), None
#             )
#
#
#         # generate caption for an image
#         def predict_caption(model, image, tokenizer, max_length):
#             # add start tag for generation process
#             in_text = 'startseq'
#             # iterate over the max length of sequence
#             for i in range(max_length):
#                 # encode input sequence
#                 sequence = tokenizer.texts_to_sequences([in_text])[0]
#                 # pad the sequence
#                 sequence = pad_sequences([sequence], max_length)
#                 # predict next word
#                 yhat = model.predict([image, sequence], verbose=0)
#                 # get index with high probability
#                 yhat = np.argmax(yhat)
#                 # convert index to word
#                 word = get_word_from_index(yhat, tokenizer)
#                 # stop if word not found
#                 if word is None:
#                     break
#                 # append word as input for generating next word
#                 in_text += " " + word
#                 # stop if we reach end tag
#                 if word == 'endseq':
#                     break
#
#             return in_text
#
#
#         def preprocess_image(image_path):
#             image = Image.open(image_path)
#             image = image.resize((224, 224))  # Resize the image to match VGG16 input size
#             image = np.array(image)
#             image = preprocess_input(image)
#             return image
#
#
#         def generate_caption_for_new_image(image_path):
#             # Preprocess the image
#             new_image = preprocess_image(image_path)
#
#             # Generate features for the new image using the pre-trained VGG16 model
#             vgg_model = VGG16()
#             vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
#             new_image_features = vgg_model.predict(np.array([new_image]), verbose=0)
#
#             predicted_caption = predict_caption(model, new_image_features, tokenizer,
#                                                 max_caption_length)
#             predicted_caption = predicted_caption.replace('startseq', '').replace('endseq', '').strip()
#             predicted_caption = predicted_caption.capitalize()
#             if not predicted_caption.endswith('.'):
#                 predicted_caption += '.'
#
#             return generate_caption_for_new_image
#


#
#         # Generate caption using the model
#         def predict_caption(model, image_features, tokenizer, max_caption_length):
#             caption = "startseq"
#             for _ in range(max_caption_length):
#                 sequence = tokenizer.texts_to_sequences([caption])[0]
#                 sequence = pad_sequences([sequence], maxlen=max_caption_length)
#                 yhat = model.predict([image_features, sequence], verbose=0)
#                 predicted_index = np.argmax(yhat)
#                 predicted_word = get_word_from_index(predicted_index, tokenizer)
#                 caption += " " + predicted_word
#                 if predicted_word is None or predicted_word == "endseq":
#                     break
#             return caption
#
#
#         # Generate caption
#         generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)
#
#         # Remove startseq and endseq
#         generated_caption = generated_caption.replace("startseq", "").replace("endseq", "")
#
#     # Display the generated caption with custom styling
#     st.markdown(
#         f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
#         f'<p style="font-style: italic;">“{generated_caption}”</p>'
#         f'</div>',
#         unsafe_allow_html=True
#     )
#
#
#
#
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
#
#
#
#
#
#
#
#
#
#     # Predict caption for the new image
#       # loaded_model, image, loaded_tokenizer
#
#     # Remove startseq and endseq tokens
#
#
#     # Capitalize the first letter
#     predicted_caption = predicted_caption.capitalize()
#
#     # Add a full stop at the end if not present
#     if not predicted_caption.endswith('.'):
#         predicted_caption += '.'
#
#     # Display the image and the predicted caption
#     plt.imshow(Image.open(image_path))
#     plt.title(predicted_caption)
#     plt.axis('off')
#     plt.show()
#
#
# # Specify the path to the new image
# new_image_path = "/content/drive/MyDrive/download.jpg"
# generate_caption_for_new_image(new_image_path)

#
# import numpy as np
# import pandas as pd
# import cv2
# import os
# from glob import glob
# import pickle
#
# import tensorflow as tf
# from tensorflow import keras
# from keras.applications.vgg16 import VGG16,preprocess_input
# #from keras.preprocessing.image import load_img,img_to_array
# from keras.preprocessing import image
#
# from keras.utils import load_img
# from keras.utils import img_to_array
#
# #from keras.preprocessing.sequence import pad_sequences
# from keras.utils import pad_sequences
#
# from keras.models import Model
#
# from keras.models import load_model
# import pickle
#
#
# # Load the model
# model = load_model("best_mode_vgg_40.h5", compile=False)
#
# # Load the tokenizer
# with open('tokenizer40.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
#
#
# def idx_to_word(integer, tokenizer):
#     for word, index in tokenizer.word_index.items():
#         if index == integer:
#             return word
#     return None
#
# # Function to preprocess an image for VGG16
# # generate caption for an image
# def predict_caption(model, image, tokenizer, max_length=35):
#     # add start tag for generation process
#     in_text = 'startseq'
#     # iterate over the max length of sequence
#     for i in range(max_length):
#         # encode input sequence
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         # pad the sequence
#         sequence = pad_sequences([sequence], max_length)
#         # predict next word
#         yhat = model.predict([image, sequence], verbose=0)
#         # get index with high probability
#         yhat = np.argmax(yhat)
#         # convert index to word
#         word = idx_to_word(yhat,tokenizer)
#         # stop if word not found
#         if word is None:
#             break
#         # append word as input for generating next word
#         in_text += " " + word
#         # stop if we reach end tag
#         if word == 'endseq':
#             break
#
#     return in_text
#
#
#
# import urllib.request
# import numpy as np
# from PIL import Image
# from keras.applications.inception_v3 import preprocess_input
#
# def load_and_preprocess_image(image_url, target_size=(299, 299)):
#     # Load image from URL
#     urllib.request.urlretrieve(image_url, 'temp_image.jpg')
#     image = Image.open('temp_image.jpg')
#
#     # Preprocess image
#     image = image.resize(target_size)
#     image = np.array(image)
#     image = preprocess_input(image)
#     image = np.expand_dims(image, axis=0)
#
#     return image




# import streamlit as st
# import numpy as np
# import pickle
# import tensorflow as tf
# from keras.models import Model
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.preprocessing.image import load_img, img_to_array
# from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
# from PIL import Image
#
# # Preprocess the uploaded image
# def preprocess_image(uploaded_image):
#     image = Image.open(uploaded_image)
#     image = image.resize((224, 224))  # Resize the image to match VGG16 input size
#     image = np.array(image)
#     image = preprocess_input(image)
#     return image
#
# # Load MobileNetV2 model
# model_1 = load_model("best_mode_vgg_40.h5", compile=False)
#
# # Load the tokenizer
# with open("tokenizer40.pickle", 'rb') as handle:
#     tokenizer = pickle.load(handle)
#
# def idx_to_word(integer, tokenizer):
#     for word, index in tokenizer.word_index.items():
#         if index == integer:
#             return word
#     return None
#
# # Function to generate captions
# def predict_caption(model, image, tokenizer, max_length):
#     # add start tag for generation process
#     in_text = 'startseq'
#     # iterate over the max length of sequence
#     for i in range(max_length):
#         # encode input sequence
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         # pad the sequence
#         sequence = pad_sequences([sequence], max_length)
#         # predict next word
#         yhat = model.predict([image, sequence], verbose=0)
#         # get index with high probability
#         yhat = np.argmax(yhat)
#         # convert index to word
#         word = idx_to_word(yhat, tokenizer)
#         # stop if word not found
#         if word is None:
#             break
#         # append word as input for generating the next word
#         in_text += " " + word
#         # stop if we reach end tag
#         if word == 'endseq':
#             break
#     return in_text
#
# # Streamlit app
# def main():
#     st.title("Image Caption Generator 📷 ➡️ 📝")
#
#     # Upload an image
#     uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
#
#     if uploaded_image is not None:
#         image = Image.open(uploaded_image)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
#
#         # Generate caption button
#         if st.button("Generate Caption"):
#             # Preprocess the uploaded image
#             new_image = preprocess_image(uploaded_image)
#
#             # Generate features for the new image using the pre-trained VGG16 model
#             vgg_model = VGG16()
#             vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
#             new_image_features = vgg_model.predict(np.array([new_image]), verbose=0)
#
#             # Predict caption for the new image
#             generated_caption = predict_caption(model_1, new_image_features, tokenizer, max_length=35)
#             generated_caption = generated_caption.replace('startseq', '').replace('endseq', '').strip()
#             generated_caption = generated_caption.capitalize()
#
#             # Display the generated caption
#             st.write("Generated Caption:", generated_caption+".")
#
# if __name__ == "__main__":
#     main()
#
#












#
# import streamlit as st
# import numpy as np
# import pickle
# import tensorflow as tf
# from keras.models import Model
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.preprocessing.image import load_img, img_to_array
# from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
# from PIL import Image
# import requests
# from io import BytesIO
#
# # Preprocess the uploaded image
# def preprocess_image(uploaded_image):
#     image = Image.open(uploaded_image)
#     image = image.resize((224, 224))  # Resize the image to match VGG16 input size
#     image = np.array(image)
#     image = preprocess_input(image)
#     return image
#
# # Preprocess the image from URL
# def preprocess_image_url(image_url):
#     response = requests.get(image_url)
#     image = Image.open(BytesIO(response.content))
#     image = image.resize((224, 224))  # Resize the image to match VGG16 input size
#     image = np.array(image)
#     image = preprocess_input(image)
#     return image
#
# # Load MobileNetV2 model
# model_1 = load_model("best_mode_vgg_40.h5", compile=False)
#
# # Load the tokenizer
# with open("tokenizer40.pickle", 'rb') as handle:
#     tokenizer = pickle.load(handle)
#
# def idx_to_word(integer, tokenizer):
#     for word, index in tokenizer.word_index.items():
#         if index == integer:
#             return word
#     return None
#
# # Function to generate captions
# def predict_caption(model, image, tokenizer, max_length):
#     # add start tag for generation process
#     in_text = 'startseq'
#     # iterate over the max length of sequence
#     for i in range(max_length):
#         # encode input sequence
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         # pad the sequence
#         sequence = pad_sequences([sequence], max_length)
#         # predict next word
#         yhat = model.predict([image, sequence], verbose=0)
#         # get index with high probability
#         yhat = np.argmax(yhat)
#         # convert index to word
#         word = idx_to_word(yhat, tokenizer)
#         # stop if word not found
#         if word is None:
#             break
#         # append word as input for generating the next word
#         in_text += " " + word
#         # stop if we reach end tag
#         if word == 'endseq':
#             break
#     return in_text
#
# # Streamlit app
# def main():
#     st.title("Image Caption Generator 📷 ➡️ 📝")
#
#     # Choose an input option: Upload or URL
#     input_option = st.radio("Select an input option:", ("Upload Image", "Image URL"))
#
#     if input_option == "Upload Image":
#         # Upload an image
#         uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
#
#         if uploaded_image is not None:
#             image = Image.open(uploaded_image)
#             st.image(image, caption="Uploaded Image", use_column_width=True)
#
#             # Generate caption button
#             if st.button("Generate Caption"):
#                 # Preprocess the uploaded image
#                 new_image = preprocess_image(uploaded_image)
#
#                 # Generate features for the new image using the pre-trained VGG16 model
#                 vgg_model = VGG16()
#                 vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
#                 new_image_features = vgg_model.predict(np.array([new_image]), verbose=0)
#
#                 # Predict caption for the new image
#                 generated_caption = predict_caption(model_1, new_image_features, tokenizer, max_length=35)
#                 generated_caption = generated_caption.replace('startseq', '').replace('endseq', '').strip()
#                 generated_caption = generated_caption.capitalize()
#
#                 # Display the generated caption
#                 st.write("Generated Caption:", generated_caption + ".")
#
#     elif input_option == "Image URL":
#         # Input image URL
#         image_url = st.text_input("Enter the image URL:")
#
#         if st.button("Generate Caption") and image_url:
#             # Preprocess the image from URL
#             new_image = preprocess_image_url(image_url)
#
#             # Generate features for the new image using the pre-trained VGG16 model
#             vgg_model = VGG16()
#             vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
#             new_image_features = vgg_model.predict(np.array([new_image]), verbose=0)
#
#             # Predict caption for the new image
#             generated_caption = predict_caption(model_1, new_image_features, tokenizer, max_length=35)
#             generated_caption = generated_caption.replace('startseq', '').replace('endseq', '').strip()
#             generated_caption = generated_caption.capitalize()
#
#                         # Display the generated caption
#             st.write("Generated Caption:", generated_caption+".")
#
# if __name__ == "__main__":
#     main()











#
# import streamlit as st
# import numpy as np
# import pickle
# import tensorflow as tf
# from keras.models import Model
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.preprocessing.image import load_img, img_to_array
# from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
# from PIL import Image
# import requests
# from io import BytesIO
#
# # Preprocess the uploaded image
# def preprocess_image(uploaded_image):
#     image = Image.open(uploaded_image)
#     image = image.resize((224, 224))  # Resize the image to match VGG16 input size
#     image = np.array(image)
#     image = preprocess_input(image)
#     return image
#
# # Preprocess the image from URL
# def preprocess_image_url(image_url):
#     response = requests.get(image_url)
#     image = Image.open(BytesIO(response.content))
#     image = image.resize((224, 224))  # Resize the image to match VGG16 input size
#     image = np.array(image)
#     image = preprocess_input(image)
#     return image
#
# # Load MobileNetV2 model
# model_1 = load_model("best_mode_vgg_40.h5", compile=False)
#
# # Load the tokenizer
# with open("tokenizer40.pickle", 'rb') as handle:
#     tokenizer = pickle.load(handle)
#
# def idx_to_word(integer, tokenizer):
#     for word, index in tokenizer.word_index.items():
#         if index == integer:
#             return word
#     return None
#
# # Function to generate captions
# def predict_caption(model, image, tokenizer, max_length):
#     # add start tag for generation process
#     in_text = 'startseq'
#     # iterate over the max length of sequence
#     for i in range(max_length):
#         # encode input sequence
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         # pad the sequence
#         sequence = pad_sequences([sequence], max_length)
#         # predict next word
#         yhat = model.predict([image, sequence], verbose=0)
#         # get index with high probability
#         yhat = np.argmax(yhat)
#         # convert index to word
#         word = idx_to_word(yhat, tokenizer)
#         # stop if word not found
#         if word is None:
#             break
#         # append word as input for generating the next word
#         in_text += " " + word
#         # stop if we reach end tag
#         if word == 'endseq':
#             break
#     return in_text
#
# # Streamlit app
# def main():
#     st.title("Image Caption Generator 📷 ➡️ 📝")
#
#     # Choose an input option: Upload or URL
#     input_option = st.radio("Select an input option:", ("Upload Image", "Image URL"))
#
#     if input_option == "Upload Image":
#         # Upload an image
#         uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
#
#         if uploaded_image is not None:
#             image = Image.open(uploaded_image)
#             st.image(image, caption="Uploaded Image", use_column_width=True)
#
#             # Generate caption button
#             if st.button("Generate Caption"):
#                 # Preprocess the uploaded image
#                 new_image = preprocess_image(uploaded_image)
#
#                 # Generate features for the new image using the pre-trained VGG16 model
#                 vgg_model = VGG16()
#                 vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
#                 new_image_features = vgg_model.predict(np.array([new_image]), verbose=0)
#
#                 # Predict caption for the new image
#                 generated_caption = predict_caption(model_1, new_image_features, tokenizer, max_length=35)
#                 generated_caption = generated_caption.replace('startseq', '').replace('endseq', '').strip()
#                 generated_caption = generated_caption.capitalize()
#
#                 # Display the generated caption
#                 st.write("Generated Caption:", generated_caption + ".")
#
#     elif input_option == "Image URL":
#         # Input image URL
#         image_url = st.text_input("Enter the image URL:")
#
#         if st.button("Generate Caption") and image_url:
#             # Preprocess the image from URL
#             new_image = preprocess_image_url(image_url)
#
#             # Generate caption button
#             if st.button("Generate Caption"):
#                 # Preprocess the uploaded image
#                 new_image = preprocess_image(image_url)
#
#             # Generate features for the new image using the pre-trained VGG16 model
#             vgg_model = VGG16()
#             vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
#             new_image_features = vgg_model.predict(np.array([new_image]), verbose=0)
#
#             # Display the image
#             image = Image.open(BytesIO(requests.get(image_url).content))
#             st.image(image, caption="Image", use_column_width=True)
#
#             # Predict caption for the new image
#             generated_caption = predict_caption(model_1, new_image_features, tokenizer, max_length=35)
#             generated_caption = generated_caption.replace('startseq', '').replace('endseq', '').strip()
#             generated_caption = generated_caption.capitalize()
#
#             # Display the generated caption
#             st.write("Generated Caption:", generated_caption + ".")
#
# if __name__ == "__main__":
#     main()


#
# import streamlit as st
# import numpy as np
# import pickle
# import tensorflow as tf
# from keras.models import Model
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.preprocessing.image import load_img, img_to_array
# from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
# from PIL import Image
# import requests
# from io import BytesIO
#
# # Preprocess the uploaded image
# def preprocess_image(uploaded_image):
#     image = Image.open(uploaded_image)
#     image = image.resize((224, 224))  # Resize the image to match VGG16 input size
#     image = np.array(image)
#     image = preprocess_input(image)
#     return image
#
# # Preprocess the image from URL
# def preprocess_image_url(image_url):
#     response = requests.get(image_url)
#     image = Image.open(BytesIO(response.content))
#     image = image.resize((224, 224))  # Resize the image to match VGG16 input size
#     image = np.array(image)
#     image = preprocess_input(image)
#     return image
#
# # Load MobileNetV2 model
# model_1 = load_model("best_mode_vgg_40.h5", compile=False)
#
# # Load the tokenizer
# with open("tokenizer40.pickle", 'rb') as handle:
#     tokenizer = pickle.load(handle)
#
# def idx_to_word(integer, tokenizer):
#     for word, index in tokenizer.word_index.items():
#         if index == integer:
#             return word
#     return None
#
# # Function to generate captions
# def predict_caption(model, image, tokenizer, max_length):
#     # add start tag for generation process
#     in_text = 'startseq'
#     # iterate over the max length of sequence
#     for i in range(max_length):
#         # encode input sequence
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         # pad the sequence
#         sequence = pad_sequences([sequence], max_length)
#         # predict next word
#         yhat = model.predict([image, sequence], verbose=0)
#         # get index with high probability
#         yhat = np.argmax(yhat)
#         # convert index to word
#         word = idx_to_word(yhat, tokenizer)
#         # stop if word not found
#         if word is None:
#             break
#         # append word as input for generating the next word
#         in_text += " " + word
#         # stop if we reach end tag
#         if word == 'endseq':
#             break
#     return in_text
#
# # Streamlit app
# def main():
#     st.title("Image Caption Generator 📷 ➡️ 📝")
#
#     # Choose an input option: Upload or URL
#     input_option = st.radio("Select an input option:", ("Upload Image", "Image URL"))
#
#     if input_option == "Upload Image":
#         # Upload an image
#         uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
#
#         if uploaded_image is not None:
#             image = Image.open(uploaded_image)
#             st.image(image, caption="Uploaded Image", use_column_width=True)
#
#             # Generate caption button
#             if st.button("Generate Caption"):
#                 # Preprocess the uploaded image
#                 new_image = preprocess_image(uploaded_image)
#
#                 # Generate features for the new image using the pre-trained VGG16 model
#                 vgg_model = VGG16()
#                 vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
#                 new_image_features = vgg_model.predict(np.array([new_image]), verbose=0)
#
#                 # Predict caption for the new image
#                 generated_caption = predict_caption(model_1, new_image_features, tokenizer, max_length=35)
#                 generated_caption = generated_caption.replace('startseq', '').replace('endseq', '').strip()
#                 generated_caption = generated_caption.capitalize()
#
#                 # Display the generated caption
#                 st.write("Generated Caption:", generated_caption + ".")
#
#     elif input_option == "Image URL":
#         # Input image URL
#         image_url = st.text_input("Enter the image URL:")
#
#         if st.button("Generate Caption") and image_url:
#             # Preprocess the image from URL
#             new_image = preprocess_image_url(image_url)
#
#             # Generate features for the new image using the pre-trained VGG16 model
#             vgg_model = VGG16()
#             vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
#             new_image_features = vgg_model.predict(np.array([new_image]), verbose=0)
#
#             # Display the image
#             image = Image.open(BytesIO(requests.get(image_url).content))
#             st.image(image, caption="Image", use_column_width=True)
#
#             # Predict caption for the new image
#             generated_caption = predict_caption(model_1, new_image_features, tokenizer, max_length=35)
#             generated_caption = generated_caption.replace('startseq', '').replace('endseq', '').strip()
#             generated_caption = generated_caption.capitalize()
#
#             # Display the generated caption
#             st.write("Generated Caption:", generated_caption + ".")
#
# if __name__ == "__main__":
#     main()







# import streamlit as st
# import numpy as np
# import pickle
# import tensorflow as tf
# from keras.models import Model
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.preprocessing.image import load_img, img_to_array
# from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
# from PIL import Image
# import requests
# from io import BytesIO
#
# # Preprocess the uploaded image
# def preprocess_image(uploaded_image):
#     image = Image.open(uploaded_image)
#     image = image.resize((224, 224))  # Resize the image to match VGG16 input size
#     image = np.array(image)
#     image = preprocess_input(image)
#     return image
#
# # Preprocess the image from URL
# def preprocess_image_url(image_url):
#     response = requests.get(image_url)
#     image = Image.open(BytesIO(response.content))
#     image = image.resize((224, 224))  # Resize the image to match VGG16 input size
#     image = np.array(image)
#     image = preprocess_input(image)
#     return image
#
# # Load MobileNetV2 model
# model_1 = load_model("best_mode_vgg_40.h5", compile=False)
#
# # Load the tokenizer
# with open("tokenizer40.pickle", 'rb') as handle:
#     tokenizer = pickle.load(handle)
#
# def idx_to_word(integer, tokenizer):
#     for word, index in tokenizer.word_index.items():
#         if index == integer:
#             return word
#     return None
#
# # Function to generate captions
# def predict_caption(model, image, tokenizer, max_length):
#     # add start tag for generation process
#     in_text = 'startseq'
#     # iterate over the max length of sequence
#     for i in range(max_length):
#         # encode input sequence
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         # pad the sequence
#         sequence = pad_sequences([sequence], max_length)
#         # predict next word
#         yhat = model.predict([image, sequence], verbose=0)
#         # get index with high probability
#         yhat = np.argmax(yhat)
#         # convert index to word
#         word = idx_to_word(yhat, tokenizer)
#         # stop if word not found
#         if word is None:
#             break
#         # append word as input for generating the next word
#         in_text += " " + word
#         # stop if we reach end tag
#         if word == 'endseq':
#             break
#     return in_text
#
# def main():
#     st.title("Image Caption Generator 📷 ➡️ 📝")
#
#     # Choose an input option: Upload or URL
#     input_option = st.radio("Select an input option:", ("Upload Image", "Image URL"))
#
#     if input_option == "Upload Image":
#         # Upload an image
#         uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
#
#         if uploaded_image is not None:
#             image = Image.open(uploaded_image)
#             st.image(image, caption="Uploaded Image", use_column_width=True)
#
#             # Generate caption button
#             if st.button("Generate Caption"):
#                 # Preprocess the uploaded image
#                 new_image = preprocess_image(uploaded_image)
#
#                 # Generate features for the new image using the pre-trained VGG16 model
#                 vgg_model = VGG16()
#                 vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
#                 new_image_features = vgg_model.predict(np.array([new_image]), verbose=0)
#
#                 # Predict caption for the new image
#                 generated_caption = predict_caption(model_1, new_image_features, tokenizer, max_length=35)
#                 generated_caption = generated_caption.replace('startseq', '').replace('endseq', '').strip()
#                 generated_caption = generated_caption.capitalize()
#
#                 # Display the generated caption
#                 st.markdown('#### Predicted Captions:')
#                 st.markdown(f"<p style='font-size:25px'><i>{generated_caption}</i>.</p>",
#                             unsafe_allow_html=True)
#
#     elif input_option == "Image URL":
#         # Input image URL
#         image_url = st.text_input("Enter the image URL:")
#
#         if image_url:
#             # Display the image
#             image = Image.open(BytesIO(requests.get(image_url).content))
#             st.image(image, caption="Image", use_column_width=True)
#
#             # Generate caption button
#             if st.button("Generate Caption"):
#                 # Preprocess the image from URL
#                 new_image = preprocess_image_url(image_url)
#
#                 # Generate features for the new image using the pre-trained VGG16 model
#                 vgg_model = VGG16()
#                 vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
#                 new_image_features = vgg_model.predict(np.array([new_image]), verbose=0)
#
#                 # Generate caption for the new image
#                 generated_caption = predict_caption(model_1, new_image_features, tokenizer, max_length=35)
#                 generated_caption = generated_caption.replace('startseq', '').replace('endseq', '').strip()
#                 generated_caption = generated_caption.capitalize()
#
#                 # Display the generated caption
#                 st.markdown('#### Predicted Captions:')
#                 st.markdown(f"<p style='font-size:25px'><i>{generated_caption}</i>.</p>",
#                             unsafe_allow_html=True)
#
# if __name__ == "__main__":
#     main()








import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from PIL import Image
import requests
from io import BytesIO
import pyttsx3
import base64


def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1500468415400-191607326b6a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80");
            background-size: 100% 100%;
            background-repeat: no-repeat;
            background-position: center center;
            width: 100%;
            height: 100%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


print(set_bg_hack_url())
#
#
# # Preprocess the uploaded image
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
with open("tokenizer40.pickle", 'rb') as handle:
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

def generate_audio(caption):
    engine = pyttsx3.init()
    engine.save_to_file(caption, 'caption_audio.mp3')
    engine.runAndWait()

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

                # Generate audio from the caption
                generate_audio(generated_caption)

                # Display the generated caption
                st.markdown('#### Predicted Caption:')
                st.markdown(f"<p style='font-size:25px'><i>{generated_caption}</i>.</p>",
                            unsafe_allow_html=True)

                # Display the audio
                st.audio('caption_audio.mp3')


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

                # Generate audio from the caption
                generate_audio(generated_caption)

                # Display the generated caption
                st.markdown('#### Predicted Caption:')
                st.markdown(f"<p style='font-size:25px'><i>{generated_caption}</i>.</p>",
                            unsafe_allow_html=True)

                # Display the audio
                st.audio('caption_audio.mp3')


if __name__ == "__main__":
    main()



