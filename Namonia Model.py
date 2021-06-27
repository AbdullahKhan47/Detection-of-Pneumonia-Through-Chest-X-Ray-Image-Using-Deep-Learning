import numpy as np
import glob
import cv2
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
 
imgaes_namonia_effected=[]
images_namonialabels_effected=[]

images_namonia_noteffected=[]
images_namonialabel_notefffected=[]

IMG_DIMS=(125,125)

files_namonia_effected=glob.glob("Your path to Pneumonia dataset")
files_namonia_noteffected=glob.glob("your path to normal dataset")

# print(files_covid_effected)
for myfiles_effected in files_namonia_effected:
    images_namonialabels_effected.append(0)
    image_namonia=cv2.imread(myfiles_effected)
    image_namonia=cv2.resize(image_namonia,dsize=IMG_DIMS,interpolation=cv2.INTER_CUBIC)
    imgaes_namonia_effected.append(image_namonia)



for myfiles_uneffected in files_namonia_noteffected:
      images_namonialabel_notefffected.append(1)
      image_notnamonia=cv2.imread(myfiles_uneffected)
      image_notnamonia=cv2.resize(image_notnamonia,dsize=IMG_DIMS,interpolation=cv2.INTER_CUBIC)
      images_namonia_noteffected.append(image_notnamonia)


files_DataFrame=pd.DataFrame({
    'Images': imgaes_namonia_effected + images_namonia_noteffected,
    'Labels': images_namonialabels_effected + images_namonialabel_notefffected
    }).sample(frac=1, random_state=42).reset_index(drop=True)

# files_DataFrame=pd.DataFrame({
#     'Images': imgaes_covid_effected + images_covid_noteffected,
#     'Labels': ["COVID EFFECTED"] * len(images_covidlabels_effected) + ["COVID NOTEFFECTED"] * len(images_covidlabel_notefffected)
#     }).sample(frac=1, random_state=42).reset_index(drop=True)

# data_top=files_DataFrame.head()
# print(data_top)
#print(files_DataFrame)

train_files, test_files, train_labels, test_labels= train_test_split(files_DataFrame['Images'].values,files_DataFrame['Labels'].values,test_size=0.2)
train_files= train_files/255
test_files= test_files/255

train_files=np.array(list(train_files))
test_files=np.array(list(test_files))    

# le=LabelEncoder()
# le.fit(train_labels)
# train_labels=le.transform(train_labels)
# test_labels=le.transform(test_labels)
   

EPOCHS=10
INPUT_SHAPE=(125,125,3)

inp=tf.keras.layers.Input(shape=INPUT_SHAPE)

conv1=tf.keras.layers.Conv2D(32,kernel_size=(3,3), activation='relu',padding='same')(inp)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
conv2= tf.keras.layers.Conv2D(64,kernel_size=(3,3), activation='relu',padding='same')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
conv3= tf.keras.layers.Conv2D(128,kernel_size=(3,3), activation='relu', padding='same')(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv3)

flat=tf.keras.layers.Flatten()(pool3)

hidden1=tf.keras.layers.Dense(512, activation='relu')(flat)
drop1=tf.keras.layers.Dropout(rate=0.3)(hidden1)
hidden2=tf.keras.layers.Dense(512, activation='relu')(drop1)
drop2=tf.keras.layers.Dropout(rate=0.3)(hidden2)
out=tf.keras.layers.Dense(1,activation='sigmoid')(drop2)

model=tf.keras.Model(inputs=inp,outputs=out)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# model.summary()

history=model.fit(train_files,train_labels,epochs=EPOCHS,validation_data=(test_files,test_labels))
model.save('SavedModel.h5')

