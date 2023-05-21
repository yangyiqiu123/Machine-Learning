import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

# from data import load_data
import numpy as np
import random

# np.random.seed(10)

# Step 1. 資料準備
# 載入資料
import pandas as pd

import os
from PIL import Image
import numpy as np

#彩色圖片輸入,將channel 1 改成 3，data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]

data = pd.read_csv('train.csv')

first_column = data.iloc[:, 0].tolist()
# print(first_column)

sec_column = data.iloc[:, 1].tolist()
# print(sec_column)

# 計算訓練資料的總數量
num_1 = len(first_column)

data_train = np.empty((num_1,1,256,256),dtype="uint8") # for train
# 代表一維的數組
label_train = np.empty((num_1,),dtype="uint8")

# print(num_1)
for i in range(num_1):
    img_1 = Image.open("./train_images/"+ first_column[i] )  # 打開圖像檔案
    print(i)
    img_1 = img_1.resize((256,256))

    arr_1 = np.array(img_1)  # 將圖像轉換成 numpy 數組

    data_train[i,:,:,:] = arr_1 # 存儲圖像數組
    
    label_train[i] = int(sec_column[i])  # 存儲圖像標籤
    

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_train, label_train, test_size=0.2)

# 將資料維度調整成模型所需的格式
x_train = x_train.transpose(0, 2, 3, 1)
x_test = x_test.transpose(0, 2, 3, 1)


# # 輸出資料集的形狀
print("train data:",'images:',x_train.shape," labels:",y_train.shape) 
print("test data:",'images:',x_test.shape," labels:",y_test.shape) 

# # 將圖片數值做歸一化，將其轉換為 0~1 之間的浮點數
x_train_normalize = x_train.astype('float32') / 255.0
x_test_normalize = x_test.astype('float32') / 255.0

# # 將標籤轉換為 one-hot 向量的形式，方便模型進行分類
y_train_OneHot = keras.utils.to_categorical(y_train)
y_test_OneHot = keras.utils.to_categorical(y_test)

# # 輸出 one-hot 標籤的形狀
print(y_train_OneHot.shape)
print(y_test_OneHot.shape)

# # Step 2. 建立模型

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

print(model.summary())


# # 載入之前訓練的模型

# # try:
# #     model.load_weights("./cifarCnnModel.h5")
# #     print("載入模型成功!繼續訓練模型")
# # except :    
# #     print("載入模型失敗!開始訓練一個新模型")


# # Step 4. 訓練模型

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


# # Step 6. 評估模型準確率

scores = model.evaluate(x_test_normalize,  y_test_OneHot, verbose=1)
print("test acc : ",scores[1])
print("test loss : ",scores[0])

train_scores = model.evaluate(x_train_normalize,  y_train_OneHot, verbose=1)
print("train acc : ",train_scores[1])
print("train loss : ",train_scores[0])


# # 進行預測

prediction=model.predict(x_test_normalize)
prediction = prediction[:10]
#print(len(prediction))
# 查看預測結果

# label_dict={0:"貴冰狗",1:"惡霸犬",2:"哈士奇",3:"柴犬",4:"吉娃娃"}
label_dict={0:"0",1:"1",2:"2",3:"3",4:"4",5:"5"}		
print(label_dict)		
for i in range(len(prediction)):
    # 返回在第 i 個預測向量中具有最大值的索引
    print("預測結果:", label_dict[np.argmax(prediction[i])])


import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')
                
        title=str(i)+','+label_dict[labels[i]]
        if len(prediction)>0:
            title+='=>'+label_dict[np.argmax(prediction[i])]
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()

plot_images_labels_prediction(x_test_normalize,y_test,prediction,0,10)

# 查看預測機率

Predicted_Probability=model.predict(x_test_normalize)

def show_Predicted_Probability(y,prediction,x_img,Predicted_Probability,i):
    print('label:',label_dict[y[i]],'predict:',label_dict[np.argmax(prediction[i])])
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(x_test[i],(256,256,1)))
    plt.show()
    for j in range(5):
        print(label_dict[j]+ ' Probability:%1.9f'%(Predicted_Probability[i][j]))

show_Predicted_Probability(y_test,prediction,x_test_normalize,Predicted_Probability,0)
show_Predicted_Probability(y_test,prediction,x_test_normalize,Predicted_Probability,3)

# Step 8. Save Weight to h5 

model.save_weights("./cifarCnnModel.h5")
print("Saved model to disk")

