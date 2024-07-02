import os
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'traffic_sign_model_cnn.h5')
model = load_model(MODEL_PATH)


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img


def getClassName(classNo):
    classes = ['Duong Cam', 'Cam di nguoc chieu', 'Cam o to', 'Cam o to re phai', 'Cam o to re trai',
               'Cam xe may', 'Cam xe tai', 'Cam o to khach va o to tai', 'Cam xe dap', 'Cam nguoi di bo',
               'Han che chieu cao xe', 'Han che chieu rong xe', 'Dung lai', 'Cam re trai', 'Cam re phai',
               'Cam quay dau', 'Cam o to quay dau', 'Toc do toi da', 'Cam bop coi', 'Cam dung va do xe',
               'Cam do xe', 'Giao nhau voi duong khong uu tien ben phai', 'Giao nhau voi duong khong uu tien ben trai',
               'Giao nhau voi duong uu tien', 'Giao nhau voi tin hieu den', 'Giao nhau voi duong sat co rao chan',
               'Giao nhau voi duong sat khong rao chan', 'Duong khong bang phang', 'Nguoi di bo cat ngang',
               'Nguy hiem tre em qua duong', 'Cong truong', 'Duong sat cat duong bo', 'Di cham',
               'Noi giao nhau chay theo vong xuyen', 'Duong danh cho nguoi di bo', 'Toc do toi thieu cho phep',
               'Het han che toc do toi thieu', 'Tuyen duong cau vuot bat qua', 'Cac xe chi duoc re trai hoac re phai',
               'Huong di vong chuong ngai vat sang trai']
    return classes[classNo]


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(32, 32))
    img = np.asarray(img)
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)[0]
    probabilityValue = np.amax(predictions)
    preds = getClassName(classIndex)
    return preds


@app.route('/', methods=['GET'])
def index():
    return "Traffic Sign Recognition API"


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'})

    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    preds = model_predict(file_path, model)
    return jsonify({'prediction': preds})


if __name__ == "__main__":
    app.run()
