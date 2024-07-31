# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:10:37 2024

@author: minec
"""

from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html', prediction=None, confidence=None)

@app.route('/', methods=['POST'])
def predict():
    #圖像輸入
    imagefile = request.files['imagefile']

    #禁用科學記號以提高清晰度
    np.set_printoptions(suppress=True)

    #加載模型
    model = load_model("keras_model_500epoch_bo_7525.h5")

    #加載標籤
    try:
        with open("labels.txt", "r", encoding='utf-8') as file:
            class_names = file.readlines()
    except UnicodeDecodeError:
        with open("labels.txt", "r", encoding='latin1') as file:
            class_names = file.readlines()

    #創建一個正確形狀的陣列來輸入到keras模型中
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    #嘗試讀取圖像，出現錯誤則進行例外處理
    try:
        image = Image.open(imagefile).convert("RGB")
    except UnidentifiedImageError:
        return render_template('index.html', prediction="nofile", confidence=None)

    #將圖像調整到至少224x224，然後從中心裁剪
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    #將圖像轉換為numpy陣列
    image_array = np.asarray(image)

    #正規化圖像
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    #將圖像加載到陣列中
    data[0] = normalized_image_array

    #使用模型來分析圖像
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip("01 ")
    confidence_score = prediction[0][index]
    confidence_score_per = format(confidence_score, ".0%")
    if "Healthy" in class_name:
        zh_tw_class_name = "健康"
    else:
        zh_tw_class_name = "有帕金森風險"

    return render_template('index.html', prediction=zh_tw_class_name, confidence=confidence_score_per)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
