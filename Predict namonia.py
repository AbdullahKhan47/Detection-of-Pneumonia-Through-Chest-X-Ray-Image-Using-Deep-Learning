import tensorflow as tf
from keras.preprocessing import image
import numpy as np

img_width,img_height=125,125

model=tf.keras.models.load_model(r'your path to the savedmodel')

sample='your path to testing image'
print(sample)
#0 label for namonia pics and 1 label for nromal pic
test_img=image.load_img(sample,target_size=(img_width,img_height))
test_img=image.img_to_array(test_img)
test_img=np.expand_dims(test_img,axis=0)
#test_img=test_img/255
result=model.predict(test_img)
print(result)
