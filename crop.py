from flask import Flask, render_template, request
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html', prediction=None, confidence=None,img_resized=None)

@app.route('/', methods=['POST'])
def predict():
    filestr = request.files['imagefile'].read()
    npimg = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use Canny edge detection
    edges = cv2.Canny(gray, 70, 210, apertureSize =5)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found, crop the region around the largest contour
    if contours:
        # Get the largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        img = img[y:y+h, x:x+w]

    # Resize and normalize the cropped image
    img_resized = cv2.resize(img, (224, 224))
    normalized_image_array = (img_resized.astype(np.float32) / 127.5) - 1

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array


    # Convert the image to Base64
    _, buffer = cv2.imencode('.png', img_resized)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    img_base64 = f"data:image/png;base64,{img_base64}"

    # 禁用科学计数法
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

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip(" 01")
    if "Healthy" in class_name:
        zh_tw_class_name="健康"
    else:
        zh_tw_class_name="有帕金森風險"

    confidence_score = prediction[0][index]

    return render_template('index.html', prediction=zh_tw_class_name, confidence=confidence_score, img_resized=img_base64)

if __name__ == '__main__':
    app.run(port=3000, debug=True)