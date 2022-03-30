import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # or any 0,1,2
import numpy as np 
import tensorflow as tf 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras_preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
        'training_data',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'test_data',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

cnn = tf.keras.models.Sequential()
#convolutional layer1
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu',input_shape=[224,224,3]))
#pooling layer1
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#convolutional layer2
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
#pooling layer2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#convolutional layer3
cnn.add(tf.keras.layers.Conv2D(filters=96, kernel_size=3, activation='relu'))
#pooling layer3
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#convolutional layer4
cnn.add(tf.keras.layers.Conv2D(filters=96, kernel_size=3, activation='relu'))
#pooling layer4
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#dropout
cnn.add(tf.keras.layers.Dropout(0.5))

#flatting
cnn.add(tf.keras.layers.Flatten())

#now we are building hidden layer for ANN
#hidden layers
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))

#output layer
cnn.add(tf.keras.layers.Dense(units=5, activation='softmax'))

#compiling our CNN
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#epochs(number of times our data is trained)
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# training_set.class_indices

#saving model
#cnn.save('my_model')