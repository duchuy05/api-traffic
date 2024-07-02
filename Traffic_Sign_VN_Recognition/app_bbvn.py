# Khai báo thư viện
import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Đường dẫn đến mô hình đã được huấn luyện
MODEL_PATH = "E:/AI/BTL/Nhom_01/traffic_sign_model_cnn.h5"

# Tải mô hình đã huấn luyện
model = load_model(MODEL_PATH)

# Hàm chuyển đổi ảnh sang ảnh xám
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# Hàm cân bằng histogram của ảnh xám
def equalize(img):
    img = cv2.equalizeHist(img)
    return img

# Hàm tiền xử lý ảnh đầu vào
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0  # Chuẩn hóa giá trị ảnh
    return img

# Hàm trả về tên của biển báo giao thông dựa vào chỉ số lớp
def getClassName(classNo):
    if classNo == 0:
        return 'Duong Cam'
    elif classNo == 1:
        return 'Cam di nguoc chieu'
    elif classNo == 2:
        return 'Cam o to'
    elif classNo == 3:
        return 'Cam o to re phai'
    elif classNo == 4:
        return 'Cam o to re trai'
    elif classNo == 5:
        return 'Cam xe may'
    elif classNo == 6:
        return 'Cam xe tai'
    elif classNo == 7:
        return 'Cam o to khach va o to tai'
    elif classNo == 8:
        return 'Cam xe dap'
    elif classNo == 9:
        return 'Cam nguoi di bo'
    elif classNo == 10:
        return 'Han che chieu cao xe'
    elif classNo == 11:
        return 'Han che chieu rong xe'
    elif classNo == 12:
        return 'Dung lai'
    elif classNo == 13:
        return 'Cam re trai'
    elif classNo == 14:
        return 'Cam re phai'
    elif classNo == 15:
        return 'Cam quay dau'
    elif classNo == 16:
        return 'Cam o to quay dau'
    elif classNo == 17:
        return 'Toc do toi da'
    elif classNo == 18:
        return 'Cam bop coi'
    elif classNo == 19:
        return 'Cam dung va do xe'
    elif classNo == 20:
        return 'Cam do xe'
    elif classNo == 21:
        return 'Giao nhau voi duong khong uu tien ben phai'
    elif classNo == 22:
        return 'Giao nhau voi duong khong uu tien ben trai'
    elif classNo == 23:
        return 'Giao nhau voi duong uu tien'
    elif classNo == 24:
        return 'Giao nhau voi tin hieu den'
    elif classNo == 25:
        return 'Giao nhau voi duong sat co rao chan'
    elif classNo == 26:
        return 'Giao nhau voi duong sat khong rao chan'
    elif classNo == 27:
        return 'Duong khong bang phang'
    elif classNo == 28:
        return 'Nguoi di bo cat ngang'
    elif classNo == 29:
        return 'Nguy hiem tre em qua duong'
    elif classNo == 30:
        return 'Cong truong'
    elif classNo == 31:
        return 'Duong sat cat duong bo'
    elif classNo == 32:
        return 'Di cham'
    elif classNo == 33:
        return 'Noi giao nhau chay theo vong xuyen'
    elif classNo == 34:
        return 'Duong danh cho nguoi di bo'
    elif classNo == 35:
        return 'Toc do toi thieu cho phep'
    elif classNo == 36:
        return 'Het han che toc do toi thieu'
    elif classNo == 37:
        return 'Tuyen duong cau vuot bat qua'
    elif classNo == 38:
        return 'Cac xe chi duoc re trai hoac re phai'
    elif classNo == 39:
        return 'Huong di vong chuong ngai vat sang trai'

# Hàm dự đoán loại biển báo từ ảnh đầu vào
def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(32, 32))  # Tải ảnh và thay đổi kích thước
    img = np.asarray(img)  # Chuyển đổi ảnh thành mảng numpy
    img = preprocessing(img)  # Tiền xử lý ảnh
    img = img.reshape(1, 32, 32, 1)  # Định dạng lại cho phù hợp với mô hình
    predictions = model.predict(img)  # Dự đoán loại biển báo
    classIndex = np.argmax(predictions, axis=1)[0]  # Lấy chỉ số lớp có xác suất cao nhất
    probabilityValue = np.amax(predictions)  # Lấy xác suất cao nhất
    preds = getClassName(classIndex)  # Lấy tên lớp từ chỉ số lớp
    return preds

# Định nghĩa route chính cho trang web
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Trả về trang index.html

# Định nghĩa route để xử lý dự đoán khi người dùng tải ảnh lên
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']  # Lấy tệp tin người dùng tải lên
        basepath = os.path.dirname(__file__)  # Đường dẫn đến thư mục hiện tại
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))  # Đường dẫn lưu tệp tin tải lên
        f.save(file_path)  # Lưu tệp tin
        preds = model_predict(file_path, model)  # Dự đoán loại biển báo
        result = preds  # Kết quả dự đoán
        return result  # Trả về kết quả dự đoán
    return None

# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(debug=True)