from flask import render_template
from flask import Flask, request, send_from_directory, render_template
from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import cv2
import random

def configure_routes(app):
    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

model = load_model("model_294400.h5")  # Model dosyan projenin kök klasöründe olmalı

# static klasörünü belirleyelim
STATIC_FOLDER = 'static'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)


def mask_face(image_path, output_path, min_hole=32, max_hole=64):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        mask_w = random.randint(min_hole, max_hole)
        mask_h = random.randint(min_hole, max_hole)
        mask_x1 = x + random.randint(0, max(1, w - mask_w))
        mask_y1 = y + random.randint(0, max(1, h - mask_h))
        mask_x2 = mask_x1 + mask_w
        mask_y2 = mask_y1 + mask_h
        cv2.rectangle(image, (mask_x1, mask_y1), (mask_x2, mask_y2), (0, 0, 0), -1)

    cv2.imwrite(output_path, image)


# Görseli [-1, 1] aralığına normalleştir
def preprocess(image):
    img = image.resize((256, 256))
    img = img_to_array(img)
    img = (img - 127.5) / 127.5
    img = np.expand_dims(img, 0)
    return img

# [-1, 1] aralığındaki görüntüyü geri çevir
def postprocess(image):
    image = (image + 1) / 2.0  # [0,1] aralığına
    image = np.clip(image[0], 0, 1)
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)



@app.route("/predict", methods=["POST", "GET"])
def predict():
    # Eğer POST ise, yeni resim yüklenmiş demektir
    if request.method == "POST" and "image" in request.files:
        file = request.files["image"]
        img = Image.open(file).convert("RGB")
        original_path = os.path.join(STATIC_FOLDER, "original.jpg")
        img.save(original_path)  # Orijinal resmi kaydet
    else:
        original_path = os.path.join(STATIC_FOLDER, "original.jpg")
        if not os.path.exists(original_path):
            return "Önce bir resim yüklemelisin.", 400

    # Her tahmin için yeniden maskele
    masked_path = os.path.join(STATIC_FOLDER, "mask.jpg")
    mask_face(original_path, masked_path)

    # Maskeleme sonrası modeli çalıştır
    img = Image.open(masked_path).convert("RGB")
    input_image = preprocess(img)
    output_image = model.predict(input_image)
    result_img = postprocess(output_image)

    result_path = os.path.join(STATIC_FOLDER, "result.jpg")
    result_img.save(result_path)

    return render_template("index.html", 
                           mask_image="mask.jpg", 
                           result_image="result.jpg",
                           original_image="original.jpg")


@app.route('/static/<filename>')
def send_image(filename):
    return send_from_directory(STATIC_FOLDER, filename)

