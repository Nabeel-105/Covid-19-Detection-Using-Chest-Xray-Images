
import cv2
import numpy as np


from keras.models import  load_model
#from keras.optimizers import SGD

def covidtest():
 img_size=50
 filepath='C:/Users/nabeel/Desktop/Fyp covid and penumonia seprate modules/covid_Detection_savedmodel with 98%'
 model=load_model(filepath,compile=True)

 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 print("model loaded :)")
 
 testimagepath="D:/Numl/fyp22/Split_covidpneumonia_Data/test/COVID19/COVID19(549).jpg"
 img = cv2.imread(testimagepath)
 img = cv2.resize(img,(img_size,img_size))
 img = np.reshape(img,[1,img_size,img_size,3])
 img = np.array(img)
 classes = model.predict_classes(img)
 return(classes)
 
 
a=covidtest()
if a==0:
    print("covid")
elif a==1:
     print("normal")

