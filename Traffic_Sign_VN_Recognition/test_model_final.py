#Khai báo thư viện
import numpy as np
import cv2
import pickle
from tf_keras.models import load_model
# -*- coding: utf-8 -*-

#############################################
#Thiết lập các hằng số và cấu hình
frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# Thiết lập camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
model = load_model("traffic_sign_model_cnn.h5")  ## rb = READ BYTE

#TIỀN XỬ LÍ HÌNH ẢNH
#Chỉnh ảnh thành xám
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

#Chỉnh độ tương phản
def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

#Chuyển đổi số lớp thành tên biển báo
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


while True:
    # ĐỌC ẢNH TỪ CAMERA
    success, imgOrignal = cap.read()

    # XỬ LÝ ẢNH
    img = np.asarray(imgOrignal)  # Chuyển ảnh từ định dạng gốc sang mảng numpy
    img = cv2.resize(img, (32, 32))  # Thay đổi kích thước ảnh về 32x32 pixel
    img = preprocessing(img)  # Tiền xử lý ảnh: chuyển sang ảnh xám, cân bằng histogram, chuẩn hóa
    cv2.imshow("Processed Image", img)  # Hiển thị ảnh đã xử lý
    img = img.reshape(1, 32, 32, 1)  # Thay đổi hình dạng ảnh để phù hợp với đầu vào của mô hình

    # HIỂN THỊ THÔNG TIN LÊN ẢNH GỐC
    cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)  # Thêm chữ "CLASS: " lên ảnh
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)  # Thêm chữ "PROBABILITY: " lên ảnh

    # DỰ ĐOÁN ẢNH
    predictions = model.predict(img)  # Dự đoán lớp của ảnh sử dụng mô hình đã huấn luyện
    classIndex = np.argmax(predictions, axis=1)  # Lấy chỉ số của lớp có xác suất cao nhất
    probabilityValue = np.amax(predictions)  # Lấy xác suất cao nhất trong các lớp dự đoán

    # HIỂN THỊ KẾT QUẢ DỰ ĐOÁN LÊN ẢNH GỐC
    cv2.putText(imgOrignal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)  # Thêm kết quả lớp dự đoán lên ảnh
    cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)  # Thêm xác suất dự đoán lên ảnh

    # HIỂN THỊ ẢNH GỐC CÓ THÊM KẾT QUẢ DỰ ĐOÁN
    cv2.imshow("Result", imgOrignal)

    # KIỂM TRA NẾU NHẤN PHÍM 'q' THÌ THOÁT VÒNG LẶP
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

#Đóng cửa sổ
cv2.destroyAllWindows()
cap.release()