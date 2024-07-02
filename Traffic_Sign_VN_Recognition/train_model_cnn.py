# Khai báo thư viện
import numpy as np
import matplotlib.pyplot as plt
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tf_keras.optimizers import Adam
from tf_keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os
import pandas as pd
from tf_keras.preprocessing.image import ImageDataGenerator

# Khai báo đường dẫn dataset và label
path = "E:/AI/BTL/API_Traffic/DataBienBaoVNFinal"
labelFile = 'E:/AI/BTL/API_Traffic/Traffic_Sign_VN_Recognition/label_final.csv'

# Cài đặt batch_size và epoch
batch_size_val = 32
epochs_val = 30
imageDimesions = (32, 32, 3)
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
            curImg = cv2.resize(curImg, (imageDimesions[0], imageDimesions[1]))
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

num_of_samples = []
cols = 5
num_classes = noOfClasses

# Tiền xử lý dữ liệu
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Tăng cường dữ liệu hình ảnh
dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=32)
X_batch, y_batch = next(batches)

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# Xây dựng mô hình huấn luyện
def myModel():
    model = Sequential() # Tạo lớp tuần tự
    # Thêm lớp Conv2D với bộ lọc là 120, size 5x5, đầu vào (32, 32, 1)
    model.add(Conv2D(120, (5, 5), input_shape=(imageDimesions[0], imageDimesions[1], 1), activation='relu'))
    model.add(Conv2D(120, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # Lớp Pooling size 2x2

    model.add(Conv2D(60, (3, 3), activation='relu'))
    model.add(Conv2D(60, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten()) # Duỗi vector 2D thành 1D
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.3)) # Giảm thiểu overfiting
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

# Huấn luyện mô hình
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                    steps_per_epoch=len(X_train) // 32, epochs=epochs_val,
                    validation_data=(X_validation, y_validation), shuffle=1)

# Biểu đồ độ mất mát
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')

# Biểu đồ độ chính xác
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

# Đánh giá mô hình
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Báo cáo phân loại
print(classification_report(y_true, y_pred_classes))

# Tính AUC cho từng lớp
roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
print('ROC AUC Score:', roc_auc)

# Lưu mô hình dạng hdf5
model.save("traffic_sign_model_cnn.h5")

