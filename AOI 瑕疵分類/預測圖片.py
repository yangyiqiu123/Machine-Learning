import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import cv2
import os


model = Sequential()

# 卷積層1與池化層1

model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(256, 256,1), 
                 activation='relu', 
                 padding='same'))

model.add(Dropout(rate=0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))

# 卷積層2與池化層2

model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))

model.add(Dropout(0.25))

model.add(MaxPooling2D(pool_size=(2, 2)))


# Step 3. 建立神經網路(平坦層、隱藏層、輸出層)

model.add(Flatten())

model.add(Dropout(rate=0.25))

model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.25))

model.add(Dense(6, activation='softmax'))


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




data = pd.read_csv('test.csv')

first_column = data.iloc[:, 0].tolist()
# print(first_column)

# sec_column = data.iloc[:, 1].tolist()
# print(sec_column)

# 計算訓練資料的總數量
num_1 = len(first_column)

data_train = np.empty((num_1,1,256,256),dtype="uint8") # for train
# 代表一維的數組
label_train = np.empty((num_1,),dtype="uint8")


ans = []

# print(num_1)
for i in range(num_1):
    img_1 = Image.open("./test_images/"+ first_column[i] )  # 打開圖像檔案
    # print(i)
    img_1 = img_1.resize((256,256))

    arr_1 = np.array(img_1)  # 將圖像轉換成 numpy 數組


    data_test = np.empty((1,1,256,256),dtype="uint8")

    # 將圖片的RGB通道分離後儲存
    data_test[0,:,:,:] = arr_1

    # 將資料轉換為神經網路所需的格式
    data_test = data_test.transpose(0, 2, 3, 1)

    # 將測試資料進行正規化
    data_test_normalize = data_test.astype('float32') / 255.0

    # 進行圖片分類預測
    prediction = model.predict(data_test_normalize)

   

    # # 定義標籤字典
    # label_dict={0:"0",1:"1",2:"2",3:"3",4:"4",5:"5"}	
    # # 

    # # 進行預測機率計算
    # print('預測結果:',label_dict[np.argmax(prediction[0])])
    predicted_label = np.argmax(prediction[0])
    
    # 将预测结果添加到列表中
    ans.append(predicted_label)



   
# 創建 DataFrame 對象
result_df = pd.DataFrame({"ID": first_column, "Label": ans})

# 將結果保存為 CSV 文件
result_df.to_csv("predictions.csv", index=False)

