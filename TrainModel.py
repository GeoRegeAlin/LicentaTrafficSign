import numpy as np
from skimage import color, transform
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_first')
import os
import glob
import cv2

def preprocess_img(img):
    hsv = color.rgb2hsv(img)
    img = color.hsv2rgb(hsv)
    img = transform.resize(img, (48, 48))
    img = np.rollaxis(img, -1)
    return img

def get_class(img_path):
    return int(img_path.split('\\')[-2])

img_dir = 'GTSRB/Training/images/'
imgs = []
labels = []
all_img_paths = glob.glob(os.path.join(img_dir, '*/*.ppm'))
np.random.shuffle(all_img_paths)
for img_path in all_img_paths:
    img = preprocess_img(cv2.imread(img_path))
    label = get_class(img_path)
    imgs.append(img)
    labels.append(label)

X = np.array(imgs, dtype='float32')
Y = np.eye(43, dtype='uint8')[labels]


def cnn_model():
    model = Sequential([Conv2D(16, (3, 3), padding='same',
                        input_shape=(3, 48, 48),
                               activation='relu'),
                        Conv2D(16, (3, 3), activation='relu'),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.5),
                        Conv2D(32, (3, 3), padding='same',
                               activation='relu'),
                        Conv2D(32, (3, 3), activation='relu'),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.5),
                        Conv2D(64, (3, 3), padding='same',
                               activation='relu'),
                        Conv2D(64, (3, 3), activation='relu'),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.5),
                        Flatten(),
                        Dense(512, activation='relu'),
                        Dropout(0.5),
                        Dense(43, activation='softmax')])
    return model


model = cnn_model()
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

batch_size = 100
epochs = 20
split_data=0.2

model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=split_data,
          callbacks=[ModelCheckpoint('CNNmodel.h5', save_best_only=True)]
          )