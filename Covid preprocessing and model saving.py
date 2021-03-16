import os
import cv2
import numpy as np
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential ,save_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten


imagepath="path_train_data"
catg=['COVID19','NORMAL']
img_size=50
Training_data=[]
def create_training_data():
    for category in catg:
        path=os.path.join(imagepath,category)
        class_num= catg.index(category)
        for img in os.listdir(path):
            try:
               img_array= cv2.imread(os.path.join(path,img))
               new_array=cv2.resize(img_array,(img_size,img_size))
               Training_data.append([new_array,class_num])              
            except Exception as e:
                print(e)
create_training_data()
print(len(Training_data))
#print(Training_data[0])

x=[]
y=[]
for features,label in Training_data:
    x.append(features)
    y.append(label)
x=np.array(x).reshape(-1,img_size,img_size,3)        
x=x/255
print(type(x))


imagepath2="path_val_data"
catg2=['COVID19','NORMAL']
img_size2=50
val_data=[]
New_val_data=[]
def create_val_data():
    for category in catg2:
        path=os.path.join(imagepath2,category)
        class_num2= catg2.index(category)
        for img in os.listdir(path):
            try:
               img_array2= cv2.imread(os.path.join(path,img))
               new_array2=cv2.resize(img_array2,(img_size2,img_size2))
               val_data.append([new_array2,class_num2])
              
               #New_Training_data=np.asarray(Training_data)
             
            except Exception as e:
                print(e)

create_val_data()
print(len(val_data))

xx=[]
yy=[]
for features,label in val_data:
    xx.append(features)
    yy.append(label)
xx=np.array(xx).reshape(-1,img_size2,img_size2,3)    
xx=xx/255
print(type(xx))

#print(Training_data[0])




model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(50,50,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x,y, batch_size=150, epochs=10, validation_data=(xx,yy))
score=model.evaluate(xx,yy, verbose=0)
print("test_loss: ",score[0])
print("test_accuracy: ",score[1])
#model_m=saving_code
modelpath='./covid_Detection_savedmodel'
save_model(model,modelpath)
