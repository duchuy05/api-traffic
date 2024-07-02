import numpy as np
import matplotlib.pyplot as plt
from tf_keras.models import Model
from tf_keras.layers import Dense, Dropout, Flatten
from tf_keras.applications import VGG16
from tf_keras.optimizers import Adam
from tf_keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import cv2
import os
import pandas as pd
from tf_keras.preprocessing.image import ImageDataGenerator

# Khai báo đường dẫn dataset và label
path = "E:/AI/BTL/Nhom_01/DataBienBaoVNFinal"
labelFile = 'E:/AI/BTL/Nhom_01/Traffic_Sign_VN_Recognition/label_final.csv'

# Cài đặt batch_size và epoch
batch_size_val = 32
epochs_val = 10
imageDimensions = (64, 64, 3)
testRatio = 0.2
validationRatio = 0.2

# Khởi tạo biến đếm và mảng để chứa hình ảnh và nhãn
count = 0
images = []
classNo = []

# Lấy danh sách các thư mục con theo path đã khai báo
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")

# Lấy dữ liệu ảnh từ các thư mục con của dataset
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        if curImg is not None:
            curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))
            images.append(curImg)
            classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

# Chia bộ dữ liệu thành các tập train, test, validation
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

print("Data Shapes")
print("Train", end="")
print(X_train.shape, y_train.shape)
print("Validation", end="")
print(X_validation.shape, y_validation.shape)
print("Test", end="")
print(X_test.shape, y_test.shape)

data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))

# Tiền xử lý dữ liệu
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi sang định dạng màu RGB
    img = cv2.resize(img, (imageDimensions[0], imageDimensions[1]))  # Resize ảnh về kích thước mong muốn
    img = img / 255.0  # Chuẩn hóa giá trị pixel về khoảng [0, 1]
    return img

# Áp dụng tiền xử lý cho từng ảnh trong tập huấn luyện, validation và test
X_train = np.array([preprocessing(img) for img in X_train])
X_validation = np.array([preprocessing(img) for img in X_validation])
X_test = np.array([preprocessing(img) for img in X_test])

# Chuyển đổi nhãn sang dạng one-hot encoding
y_train = to_categorical(y_train, num_classes=noOfClasses)
y_validation = to_categorical(y_validation, num_classes=noOfClasses)
y_test = to_categorical(y_test, num_classes=noOfClasses)

# Tạo mô hình VGG16
def create_vgg16_model(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Đóng băng các layer của VGG16
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Tạo và compile mô hình
model = create_vgg16_model(input_shape=imageDimensions, num_classes=noOfClasses)
print(model.summary())

# Huấn luyện mô hình
history = model.fit(X_train, y_train, batch_size=batch_size_val, epochs=epochs_val,
                    validation_data=(X_validation, y_validation))

# Biểu đồ độ mất mát
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

# Biểu đồ độ chính xác
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Đánh giá mô hình trên tập test
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# Dự đoán và tính các metrics
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Báo cáo phân loại
print(classification_report(y_true, y_pred_classes))

# Tính AUC cho từng lớp
roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
print('ROC AUC Score:', roc_auc)

# Lưu mô hình
model.save("traffic_sign_model_vgg16.h5")
