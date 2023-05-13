import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import cv2
import os

# name=input("input picture name : ")
inputname="b.jpg"

if os.path.isfile(inputname):  # 檢查檔案是否存在
    img = cv2.imread(inputname)
    new_width, new_height = 32, 32 # 設定新的寬度和高度
    img_resized = cv2.resize(img, (new_width, new_height))   # 修改圖片大小
    cv2.imwrite('output.jpg', img_resized)  # 將修改後的圖片保存為原始檔案名稱
    
        
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

try:
    model.load_weights("./cifarCnnModel.h5")
    print("success")
except:
    print("error")

# def plot_image(image):
#     fig=plt.gcf()
#     fig.set_size_inches(2, 2)
#     plt.imshow(image,cmap='binary')
#     plt.show()


inputimg = np.array(Image.open(inputname))
# img = np.array(Image.open('110321005.jpg')) 

# 顯示測試圖片
plot_image(inputimg)

img = np.array(Image.open("output.jpg"))
# 建立空的3D Numpy陣列，儲存圖片
data_test = np.empty((1,3,32,32),dtype="uint8")

# 將圖片的RGB通道分離後儲存
data_test[0,:,:,:] = [img[:,:,0],img[:,:,1],img[:,:,2]]

# 將資料轉換為神經網路所需的格式
data_test = data_test.transpose(0, 2, 3, 1)

# 將測試資料進行正規化
data_test_normalize = data_test.astype('float32') / 255.0

# 進行圖片分類預測
prediction = model.predict(data_test_normalize)

# 取出前10個預測結果
prediction = prediction[:5]

# 定義標籤字典
# label_dict={0:"0",1:"1",2:"2",3:"3",4:"4"}	
label_dict={0:"貴冰狗",1:"惡霸犬",2:"哈士奇",3:"柴犬",4:"吉娃娃"}
# 

# 進行預測機率計算
Predicted_Probability = model.predict(data_test_normalize)

# 定義顯示預測結果與機率的函數
def show_Predicted_Probability(prediction, x_img, Predicted_Probability):
    # 顯示預測結果
    print('預測結果:',label_dict[np.argmax(prediction[0])])
    # 顯示測試圖片
    # plt.figure(figsize=(2,2))
    # plt.imshow(np.reshape(data_test[0],(32, 32,3)))
    # 顯示每個類別的預測機率
    for j in range(5):
        print(label_dict[j]+ '%1.9f'%(Predicted_Probability[0][j]))

# 顯示第一個測試圖片的預測結果與機率
show_Predicted_Probability(prediction, data_test, Predicted_Probability)
