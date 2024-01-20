import numpy as np
import pickle
import cv2
from PIL import Image
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
import os
import matplotlib.pyplot as plt

#finds the number of images in the healthy training set
num = len(os.listdir("dataset/drawings/spiral/training/healthy"))
data_train = []
data_train_num = []
#opens the healthy training data set and adds each image into a list "data_train"
#adds '0' to the list "data_train_num" since all the images will be classified as healthy
for i in range(num):
    img = Image.open("dataset/drawings/spiral/training/healthy/"+ os.listdir("dataset/drawings/spiral/training/healthy")[i])
    data_train.append(img)
    data_train_num.append(0)

#finds the number of images in the parkinson training set
num = len(os.listdir("dataset/drawings/spiral/training/parkinson"))
#opens the parkinson training data set and adds each image to the same list as above "data_train"
#adds '1' to the same list as above "data_train_num" since all the images will be classified as parkinson
for i in range(num):
    img = Image.open("dataset/drawings/spiral/training/parkinson/"+ os.listdir("dataset/drawings/spiral/training/parkinson")[i])
    data_train.append(img)
    data_train_num.append(1)
#converts the "data_train" list to a numpy array and assigns it to "x_train" --> this contains both the healthy and parkinson images
x_train = np.array(data_train,dtype=object)
#converts the "data_train_num" list to a numpy array and assigns it to "y_train" --> this contains the diagnoses of '0' (healthy) or '1' (parkinson)
y_train = np.array(data_train_num)
print(x_train.shape)
print(y_train.shape)


#does the same as the code above except with the test set
num = len(os.listdir("dataset/drawings/spiral/testing/healthy"))
data_test = []
data_test_num = []
for i in range(num):
    img = Image.open("dataset/drawings/spiral/testing/healthy/"+ os.listdir("dataset/drawings/spiral/testing/healthy")[i])
    data_test.append(img)
    data_test_num.append(0)

num = len(os.listdir("dataset/drawings/spiral/testing/parkinson"))
for i in range(num):
    img = Image.open("dataset/drawings/spiral/testing/parkinson/"+ os.listdir("dataset/drawings/spiral/testing/parkinson")[i])
    data_test.append(img)
    data_test_num.append(1)

x_test = np.array(data_test,dtype=object)
y_test = np.array(data_test_num)
print(x_test.shape)
print(y_test.shape)



#edit this

#plt.figure(figsize=(20, 10))
#sns.barplot(data = data_train).set_title("Number of training images per category:")
#plt.show()


train_data_generator = ImageDataGenerator(rotation_range=0, 
                                    width_shift_range=0.0, 
                                    height_shift_range=0.0, 
#                                     brightness_range=[0.5, 1.5],
                                    horizontal_flip=True, 
                                    vertical_flip=True)

#converts "x_train" and "y_train" back to a list format as x and y
x = list(x_train)
y = list(y_train)

x_aug_train = []
y_aug_train = []

for (i, v) in enumerate(y):
    x_img = x[i]
    x_img = np.array(x_img)
    x_img = np.expand_dims(x_img, axis=0)
    aug_iter = train_data_generator.flow(x_img, batch_size=1, shuffle=True)
    for j in range(70):
        aug_image = next(aug_iter)[0].astype('uint8')
        x_aug_train.append(aug_image)
        y_aug_train.append(v)
print(len(x_aug_train))
print(len(y_aug_train))

x_train = x + x_aug_train
y_train = y + y_aug_train
print(len(x_train))
print(len(y_train))

test_data_generator = ImageDataGenerator(rotation_range=0, 
                                    width_shift_range=0.0, 
                                    height_shift_range=0.0, 
#                                     brightness_range=[0.5, 1.5],
                                    horizontal_flip=True, 
                                    vertical_flip=True)

x = list(x_test)
y = list(y_test)

x_aug_test = []
y_aug_test = []

for (i, v) in enumerate(y):
    x_img = x[i]
    x_img = np.array(x_img)
    x_img = np.expand_dims(x_img, axis=0)
    aug_iter = test_data_generator.flow(x_img, batch_size=1, shuffle=True)
    for j in range(20):
        aug_image = next(aug_iter)[0].astype('uint8')
        x_aug_test.append(aug_image)
        y_aug_test.append(v)
print(len(x_aug_test))
print(len(y_aug_test))

x_test = x + x_aug_test
y_test = y + y_aug_test
print(len(x_test))
print(len(y_test))

for i in range(len(x_train)):
    img = x_train[i]
    img = np.array(img) 
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_train[i] = img
    
for i in range(len(x_test)):
    img = x_test[i]
    img = np.array(img) 
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_test[i] = img
        
x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = x_train/255.0
x_test = x_test/255.0

label_encoder = LabelEncoder()
#y_train = np.array(y_train, dtype=object) 
y_train = label_encoder.fit_transform(y_train)
print(y_train.shape)

label_encoder = LabelEncoder()
#y_test = np.array(y_test, dtype=object) 
y_test = label_encoder.fit_transform(y_test)
print(y_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

def parkinson_disease_detection_model(input_shape=(224, 224, 1)):
    #regularizer = tf.keras.regularizers.l2(0.001)
    _input = Input((224,224,1)) 
    #model = Sequential()
    #model.add(Input(shape=input_shape))
    conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(_input)
    conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
    pool1  = MaxPooling2D((2, 2))(conv2)

    conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
    conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
    pool2  = MaxPooling2D((2, 2))(conv4)

    conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
    conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
    conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
    pool3  = MaxPooling2D((2, 2))(conv7)

    conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
    conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
    conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
    pool4  = MaxPooling2D((2, 2))(conv10)

    conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
    conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
    conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
    pool5  = MaxPooling2D((2, 2))(conv13)

    flat   = Flatten()(pool5)
    dense1 = Dense(4096, activation="relu")(flat)
    dense2 = Dense(4096, activation="relu")(dense1)
    output = Dense(2, activation="softmax")(dense2)

    model  = Model(inputs=_input, outputs=output)
    optimizer = Adam(3.15e-5)
    #opt = Adam(learning_rate=0.001)
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model= parkinson_disease_detection_model(input_shape=(224, 224, 1))
model.summary()

hist = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test))

figure = plt.figure(figsize=(10, 10))
plt.plot(hist.history['accuracy'], label='Train_accuracy')
plt.plot(hist.history['val_accuracy'], label='Test_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")
plt.show()

figure2 = plt.figure(figsize=(10, 10))
plt.plot(hist.history['loss'], label='Train_loss')
plt.plot(hist.history['val_loss'], label='Test_loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper left")
plt.show()



ypred = model.predict(x_test)
ypred = np.argmax(ypred, axis=1)
y_test_pred = np.argmax(y_test, axis=1)
print(classification_report(y_test_pred, ypred))

matrix = confusion_matrix(y_test_pred, ypred)
df_cm = pd.DataFrame(matrix, index=[0, 1], columns=[0, 1])
figure = plt.figure(figsize=(5, 5))
sns.heatmap(df_cm, annot=True, fmt='d')

model_pkl_file = "parkinson_spiral1.pkl"  

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(model, file)
