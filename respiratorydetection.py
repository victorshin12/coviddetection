import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Dropout,Activation,Add,MaxPooling2D,Conv2D,Flatten
from keras.models import Sequential 
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from tensorflow.keras.preprocessing import image


import os
for dirname, _, filenames in os.walk('/content/drive/MyDrive/COVID-19_Radiography_Dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#processing the data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/MyDrive/COVID-19_Radiography_Dataset',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(244, 244),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear",
    follow_links=False,
)
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    '/content/drive/MyDrive/COVID-19_Radiography_Dataset',
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(244, 244),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear",
    follow_links=False,
)

#classes 
class_names =  ["COVID-19","NORMAL","Viral Pneumonia"]
for i in range(len(class_names)):
    print(class_names[i] ," " , i)

#visualize data
image_path = "/content/drive/MyDrive/COVID-19_Radiography_Dataset/COVID/COVID-634.png"
new_img = image.load_img(image_path, target_size=(244, 244))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
plt.title("COVID-19", fontsize=20)
plt.imshow(new_img)

model = tf.keras.models.Sequential([
  layers.BatchNormalization(),
  
  layers.Conv2D(8, 3, activation='relu'),

  layers.Dropout(0.3),
  layers.MaxPooling2D(),
  layers.Conv2D(16, 3, activation='relu'),

  layers.Dropout(0.3),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),

  layers.Flatten(),
  layers.Dense(32, activation='relu'),
  layers.Dense(3, activation= 'softmax')
])

early = tf.keras.callbacks.EarlyStopping(monitor ='val_loss', patience = 5)
model.compile(optimizer = 'RMSprop',loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history=model.fit(train_data, validation_data = test_data,batch_size = 32, epochs = 10, callbacks=[early])

#plotting training values
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

#example 1
image_path = "/content/drive/MyDrive/COVID-19_Radiography_Dataset/Normal/Normal-120.png"
new_img = image.load_img(image_path, target_size=(244, 244))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
print(prediction)
prediction = np.argmax(prediction,axis=1)
print(prediction)
print(class_names[prediction[0]])
plt.imshow(new_img)

from tensorflow.keras.models import Model

base_model = model
activation_model = Model(inputs=base_model.inputs, outputs=base_model.layers[1].output)

img_path = '/content/drive/MyDrive/COVID-19_Radiography_Dataset/COVID/COVID-910.png'
img = image.load_img(img_path, target_size=(244, 244))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

activation = activation_model(img_tensor)

plt.figure(figsize=(20,20))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(activation[0,:,:,i])
plt.show()
