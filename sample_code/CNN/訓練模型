# coding: utf-8

import cv2
import os
from PIL import Image
import numpy as np

def load_data():
    num_1 = len(os.listdir("./trainImg"))
    num_2 = len(os.listdir("./testImg"))

    train_data=[]
    train_labels=[]
    for img_file in os.listdir('./trainImg'):
        img=cv2.imread('trainImg/' + img_file)
        train_data.append(img)
        image_id=int(img_file.split('.',1)[0])
        train_labels.append(image_id)


    test_data=[]
    test_labels=[]
    for img_file in os.listdir('./testImg'):
        img=cv2.imread('testImg/' + img_file)
        test_data.append(img)
        image_id=int(img_file.split('.',1)[0])
        test_labels.append(image_id)
    
    train_data=np.array(train_data)
    train_labels=np.array(train_labels)
    test_data=np.array(test_data)
    test_labels=np.array(test_labels)

    return (train_data,train_labels), (test_data,test_labels)





# from google.colab import files
# uploaded = files.upload()
# import sys
# sys.path.append('/content/drive/MyDrive/mlproject/ML')

import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

# from data import load_data
import numpy as np
import random

np.random.seed(10)

# Step 1. 資料準備
# 載入資料
print("Loading data...")
(x_train,y_train),(x_test,y_test)=load_data()  # load_data() 函式的實現需要從其他地方引入
print("Data loaded")

# 將資料維度調整成模型所需的格式
# x_train = x_train.transpose(0, 2, 3, 1)
# x_test = x_test.transpose(0, 2, 3, 1)

# 將訓練集的圖片和標籤互相打亂，以增加隨機性
index_1 = [i for i in range(len(x_train))]
random.shuffle(index_1)
x_train = x_train[index_1]
y_train = y_train[index_1]

# 將測試集的圖片和標籤互相打亂，以增加隨機性
index_2 = [i for i in range(len(x_test))]
random.shuffle(index_2)
x_test = x_test[index_2]
y_test = y_test[index_2]

# 輸出資料集的形狀
# print("train data:",'images:',x_train.shape," labels:",y_train.shape) 
# print("test data:",'images:',x_test.shape," labels:",y_test.shape) 

# 將圖片數值做歸一化，將其轉換為 0~1 之間的浮點數
x_train_normalize = x_train.astype('float32') / 255.0
x_test_normalize = x_test.astype('float32') / 255.0

# 將標籤轉換為 one-hot 向量的形式，方便模型進行分類
y_train_OneHot = keras.utils.to_categorical(y_train)
y_test_OneHot = keras.utils.to_categorical(y_test)

# 輸出 one-hot 標籤的形狀
# print(y_train_OneHot.shape)
# print(y_test_OneHot.shape)

# Step 2. 建立模型

model = Sequential()

# 卷積層1與池化層1

model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(32, 32,3), 
                 activation='relu', 
                 padding='same'))

model.add(Dropout(rate=0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷積層2與池化層2

model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 activation='relu', padding='same'))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 activation='relu', padding='same'))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))


# Step 3. 建立神經網路(平坦層、隱藏層、輸出層)

model.add(Flatten())

model.add(Dropout(rate=0.25))

model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.25))


model.add(Dense(5, activation='softmax'))
print(model.summary())




# Step 4. 訓練模型

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc'])
train_history=model.fit(x_train_normalize, y_train_OneHot,
                        validation_split=0.2,
                        epochs=15, batch_size=32, verbose=1)          

import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')
show_train_history('loss','val_loss')


# Step 6. 評估模型準確率

scores = model.evaluate(x_test_normalize,  y_test_OneHot, verbose=1)
print("test acc : ",scores[1])
print("test loss : ",scores[0])

train_scores = model.evaluate(x_train_normalize,  y_train_OneHot, verbose=1)
print("train acc : ",scores[1])
print("train loss : ",scores[0])

# Step 8. Save Weight to h5 

model.save_weights("./cifarCnnModel.h5")
print("Saved model to disk")
