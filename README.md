<p align="center">
<img src="https://img.shields.io/badge/Made%20by%20-Abdur Sobhan Riad-blue">
<img src="https://badges.frapsoft.com/os/v1/open-source.svg?v=103">
<img src="https://img.shields.io/badge/Contributions-welcome-brightgreen">
<img src="https://img.shields.io/badge/python-v3.9%2B-orange" />
<img src="https://img.shields.io/badge/tensorflow-2.13.0-yellow">
<a href="https://www.linkedin.com/in/md-abdur-sobhan-riad-360b521b1/">
  <img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn Profile">
</a>
</p>

<h1> Image_Caption_Generator </h1>


<p align="justify">
This is a Deep Learning Model for generating captions for images. It uses techniques from Computer Vision and Natural Language Processing. Some examples of images from test dataset and the captions generated by the model are shown below.
</p>

<p align="center">
  <img alt="Image Captions demo"  
       src="demo/ezgif-1-ba79ce21f2.gif"
       height="300">
</p>


<h3> TABLE OF CONTENTS </h3>
<ol>
    <li><a href="#intro"> Introduction </a></li>
    <li><a href="#dataset"> Dataset </a></li>
    <li><a href="#model"> Used Models </a> </li>
    <li><a href="#performance"> Models performance </a> </li>
    <li><a href="#result"> Predicted Result </a> </li>
<!--     <li><a href="#examples"> Examples </a></li> -->
    <li><a href="#frameworks"> Frameworks, Libraries & Languages </a></li>
    <li><a href="#deployment"> Deployment </a></li>
    <li><a href="#demo"> Demo </a></li>
    <li><a href="#conclusion"> Conclusion </a></li>
    <li><a href="#acknowledgement"> Acknowledgement </a></li>
</ol>

<h2 id="intro"> Introduction </h2>
<p align="justify">
Deep Learning and Neural Networks have found profound applications in both NLP and Computer Vision. Before the Deep Learning era, statistical and Machine Learning techniques were commonly used for these tasks, especially in NLP. Neural Networks however have now proven to be powerful techniques, especially for more complex tasks. With the increase in size of available datasets and efficient computational tools, Deep Learning is being throughly researched on and applied in an increasing number of areas.Image captioning is the process of taking an image and generating a caption that accurately describes the scene. This is a difficult task for neural networks because it requires understanding both natural language and computer vision. It is intersaction between NLP and Computer Vision.
<p align="center">
<br><br><img src="demo/Screenshot_89.png" width=700 height=400>
</p>
<br>
In 2012 the Google <a href="http://www.image-net.org/">ImageNet</a> Challenge (ILSVRC) results showed that Convolutional Neural Networks (CNNs) can be an excellent choice for tasks involving visual imagery. Being translation invariant, after learning a pattern in one region of an image, CNNs can very easily recognize it in another region - a task which was quite computationally inefficient in vanilla feed-forward networks. When many convolutional layers are stacked together, they can efficiently learn to recognize patterns in a hierarchical manner - the initial layers learn to detect edges, lines etc. while the later layers make  use of these to learn more complex features. In this project, we make use of a popular CNN architecture - the ResNet50 to process the input images and get the feature vectors.
<br>
For generating the captions, we make use of Long Short-Term Memory (LSTM) networks. LSTMs are a variant of Recurrent Neural Networks which are widely used in Natural Language Processing. Unlike a Dense layer, an RNN layer does not process an input in one go. Instead, it processes a sequence element-by-element, at each step incorporating new data with the information processed so far. This property of an RNN makes it a natural yet powerful architecture for processing sequential inputs.
</p>

<h2 id="dataset"> Dataset </h2>

This project uses the ```Flickr 8K``` dataset for training the model. This can be downloaded from <a href="https://www.kaggle.com/shadabhussain/flickr8k?select=model_weights.h5"> here</a>. It contains 8000 images, most of them featuring people and animals in a state of action.Though i have used two more dataset like [Flickr 30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) and [Microsoft CoCo](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) but flickr 8k comparatively doing well from among datasets.Here Each image is provided with five different captions describing the entities and events depicted in the image. Different captions of the same image tend to focus on different aspects of the scene, or use different linguistic constructions. This ensures that there is enough linguistic variety in the description of the images.

</p>

<h2 id="model"> Model </h2>

<p align="justify">
    This project i used several models like vgg16 with lstm,resnet50 with lstm, inception v3 wiht lstm but considering everything i deployed vgg16 with lstm for my convenience the Vgg16 architecture for obtaining the image features. VGG networks, standing for Visual Geometry Group networks, have played a pivotal role in the realm of Computer Vision. Their significance was highlighted when they secured victory in the prestigious ImageNet Challenge. VGG networks demonstrated a groundbreaking concept—how even deep neural networks (VGG16 and VGG19 are examples) can be effectively trained, transcending the limitations of the vanishing gradient problem.

The hallmark of VGG networks is their simplicity and uniformity. Unlike the original ResNet with its 152 layers, VGG networks feature a straightforward and easily understandable architecture. This simplicity, combined with their effectiveness, makes them a popular choice for various computer vision tasks.

In the world of Transfer Learning, VGG networks shine brightly. Keras, a widely-used deep learning framework, includes pre-trained VGG models along with weights fine-tuned on the expansive ImageNet dataset. This availability has made VGG networks a go-to choice for researchers and practitioners in the field. Since we only need this network for getting the image feature vectors, so we remove the last layer (which in the original model was used to classify input image into one of the 1000 classes). The encoded features for training and test images are stored at "encoded_train_features.pkl" and "encoded_test_features.pkl" respectively.
</p>

<h2 id="performance"> Models performance </h2>

Models | Accuracy | BLEU-1 | BLEU-1
--- | --- | --- | ---
Vgg 16 with Lstm | 0.5125 | 0.534878 | 0.308821
Resnet 50 with Lstm | 0.5522 | 0.548153 | 0.321559
InceptionV3 with Lstm | 0.5012 | 0.545291 | 0.323035
