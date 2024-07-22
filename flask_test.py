# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:10:37 2024

@author: minec
"""

from flask import Flask, render_template, request
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

app=Flask(__name__)

@app.route('/',methods=['Get'])
def hello_world():
    return render_template('index.html', prediction=None, confidence=None)

@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model_200epoch.h5")

    # Load the labels
    try:
        with open("labels.txt", "r", encoding='utf-8') as file:
            class_names = file.readlines()
    except UnicodeDecodeError:
        with open("labels.txt", "r", encoding='latin1') as file:
            class_names = file.readlines()

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return render_template('index.html', prediction=class_name, confidence=confidence_score)



if __name__=='__main__':
    app.run(port=3000,debug=True)