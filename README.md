# plant-leaf-disease-detection
from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
from keras.preprocessing.image
import ImageDataGenerator, img_to_array, load_img from keras.applications.vgg19
import VGG19, preprocess_input, decode_predictions

train_datagen = ImageDataGenerator(zoom_range = 0.5, shear_range = 0.3, horizontal_flip=True, preprocessing_function = preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function= preprocess_input )

train=train_datagen.flow_from_directory(directory="/content/drive/MyDrive/Dataset/train",target_size=(256,256),batch_size=32)
val=val_datagen.flow_from_directory(directory="/content/drive/MyDrive/Datasetarget_
size=(256,256),batch_size=32)

def plotImage(img_arr, lable):
for im , l in zip(img_arr , lable):
plt.figure(figsize=(5,5))
plt.imshow(im)
plt.show()

plotImage(t_img[:3], lable[:3])

from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19, preprocess_input
import keras

base_model = VGG19(input_shape=(256,256,3), include_top = False)
for layer in base_model.layers:
layer.trainable = False
base_model.summary()


X = Flatten()(base_model.output)
X = Dense(units = 10, activation ='softmax')(X)

model = Model(base_model.input, X)
model.summary()
model.compile(optimizer= 'adam'
loss= keras.losses.categorical_crossentropy, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStoppinges = EarlyStopping(monitor = 'val_accuracy', min_delta= 0.01, patience=3 , verbose=1)

mc = ModelCheckpoint(filepath="best_model.h5",
monitor = 'val_accuracy',
min_delta= 0.01, patience=3 ,
verbose=1,
save_best_only= True)

his = model.fit(x= train,steps_per_epoch=1,
epochs = 12,
verbose= 1,
callbacks= cb,
validation_data=val,
validation_steps=1)

h = his.history
h.keys()

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c = "red")
plt.title("acc vs v.acc")
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'], c = "red")
plt.title("loss vs v.loss")
plt.show()

from keras.models import load_model
model = load_model("/content/best_model.h5")
acc = model.evaluate_generator(val)[1]
print(f"The accuracy of given model is = {acc*100}%")
ref = dict(zip(list(train.class_indices.values()),list(train.class_indices.keys()))

def prediction(path):
img = load_img(path, target_size =(256,256))
i = img_to_array(img)
im = preprocess_input(i)
img = np.expand_dims(im , axis= 0)
pred = np.argmax(model.predict(img) )
print(f" the disease is { ref[pred] }")

path = "/content/drive/MyDrive/Dataset/test/Tomato Septoria_leaf_spot (1).JPG"
prediction(path
