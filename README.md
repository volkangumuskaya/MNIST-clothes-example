# MNIST-clothes-example
Toy example to try out CV in my local . Maybe I will make a repo that will be executable with GitHub workflow in the future. Below is just for illustration and copy/paste purposes. 

Anyway.. Training the model is simple using online sources. Had to play around array size while testing the model with own images created by screenshots saved in my local. Could not get a decent accuracy though, but this was not the main purpose anyway.

```
#remove all variables
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

for name in dir():
    if not name.startswith('_'):
        del locals()[name]
```

Import libraries

```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```
Import dataset 
```
#Import mnist dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
```
Normalize for training 
```
#normalize
training_images  = training_images / 255.0
test_images = test_images / 255.0
print(training_images.shape)
print(test_images.shape)
```
Build the model architecture
```
#build and compile the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = tf.keras.optimizers.Adam(),loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
```

Actual training of the model

```
#train the model
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
print(model.evaluate(test_images, test_labels))
```

Make predcitions and get accuracy score
```
#get predictions
classifications = model.predict(test_images)

#get labels from predcitions
predictions=np.argmax(classifications, axis=-1)

#get accuracy manually (same as model.evaluate(...))
(test_labels==predictions).sum()/len(test_labels)
```

Labels list according to https://developers.google.com/codelabs/tensorflow-2-computervision#1
|Name | Description |
|-----|-------------|
0 |T-shirt/top
1 |Trouser
2 |Pullover
3 |Dress
4 |Coat
5 |Sandal
6 |Shirt
7 |Sneaker
8 |Bag
9 |Ankle boot

## Below is the code to make predcitions using own images
The images can be found here: https://github.com/volkangumuskaya/images_for_CV_example


```
#####Test on custom images I got from google images, saved as png in [custom_tests]
from keras.preprocessing import image
import keras
import os
import cv2

#Set directory names
dir_path=os.getcwd()
# Directory with our training horse pictures
custom_test_folder = os.path.join(dir_path,'custom_tests\\')

for file_counter in range(1,11):

    #Make test on an image
    filename='test'+str(file_counter)+'.png'
    img_width, img_height = 28, 28

    #load image and set to the same size as model expects
    print("Predcting for ",filename)
    img = image.load_img(custom_test_folder+filename, target_size = (img_width, img_height))
    img = image.img_to_array(img)
    # print(img.shape)
    #change channel to 1 because model requires that
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img.shape)

    #Alternative method to change channel to 1 (BRG to black and white)
    # img = image.load_img(custom_test_folder+filename, target_size = (img_width, img_height),color_mode='grayscale')
    # img=(np.squeeze(img))
    # img.shape

    img = np.expand_dims(img, axis = 0)

    #normalize
    img=img/255
    img=1-img

    #make predcition for the single image and print label (gives 8 for any of the pictures i tried
    print("Predicted label is ",np.argmax(model.predict(img), axis=-1), "for file:",filename)
    print("Just a check to see if img array is identical in all cases\n" ,img[0,0,0:5])
    del(img)
```
