import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense,Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from matplotlib.image import imread
import numpy as np


# dataset Path

train = r'D:\CNN\GUI app for tomato leafs disease classification using CNN (1)\GUI app for tomato leafs disease classification using CNN\archive (1)\tomato\train'
test = r'D:\CNN\GUI app for tomato leafs disease classification using CNN (1)\GUI app for tomato leafs disease classification using CNN\archive (1)\tomato\val'

trainDIR = r'D:\CNN\GUI app for tomato leafs disease classification using CNN (1)\GUI app for tomato leafs disease classification using CNN\archive (1)\tomato\train'
testDIR = r'D:\CNN\GUI app for tomato leafs disease classification using CNN (1)\GUI app for tomato leafs disease classification using CNN\archive (1)\tomato\val'

# data Size

size = 224
batch_size = 32
epoch = 10


# Data Augmentation

datagen = ImageDataGenerator(rescale=1./255, shear_range= 0.2,zoom_range=0.2,horizontal_flip=True,validation_split=0.2)

X_train = datagen.flow_from_directory(train,target_size= (size,size),batch_size=batch_size, class_mode='categorical',subset='training')

X_test = ImageDataGenerator(rescale=1./255).flow_from_directory(test,target_size= (size,size),batch_size=batch_size, class_mode='categorical',subset='training')

X_test.class_indices.keys()


# call back setup

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

checkpoint = ModelCheckpoint(r'PlantLeafCNN_2023-04-06_final_VGG16.h5',monitor = 'val_loss',mode='min',save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss',min_delta=0,
                          patience= 20,verbose=1,restore_best_weights= True)

callbacks= [checkpoint,earlystop]


# CNN Model

model= Sequential()
model.add(Conv2D(32,(3,3),activation ='relu', padding='same',input_shape=(size,size,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(16,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(8,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(8,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(8,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(126,activation='relu'))
model.add(Dense(3,activation='softmax'))


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# print model summary

history = model.fit(x=X_train, validation_data=X_test,
                    epochs= epoch,
                    steps_per_epoch=X_train.samples//batch_size,
                    validation_steps=X_test.samples//batch_size,
                    callbacks=callbacks)

# history

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1,11)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


