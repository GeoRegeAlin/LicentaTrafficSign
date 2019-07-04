from keras.models import load_model
from keras.preprocessing import image
from skimage import color, transform
import numpy as np
import json
import cv2

def preprocess_img(img):
    hsv = color.rgb2hsv(img)
    img = color.hsv2rgb(hsv)
    img = transform.resize(img, (48, 48))

    img = np.rollaxis(img, -1)

    return img

def classification(id):
    with open('IdentitateSemn.txt') as json_file:
        data=json.load(json_file)
    print(data[str(id)])


model = load_model("CNNmodel.h5")
new_image = image.load_img("Examples/Cedeaza.jpg")
new_image=preprocess_img(new_image)
image = np.expand_dims(new_image, axis=0)
pred = model.predict(image)
print(classification(str(np.argmax(pred)+1)))

new_image = cv2.imread("Examples/Limita30km.jpg")
new_image=preprocess_img(new_image)
image = np.expand_dims(new_image, axis=0)
pred = model.predict(image)
print(classification(str(np.argmax(pred)+1)))

new_image = cv2.imread("Examples/Limita20km.jfif")
new_image=preprocess_img(new_image)
image = np.expand_dims(new_image, axis=0)
pred = model.predict(image)
print(classification(str(np.argmax(pred)+1)))

new_image = cv2.imread("Examples/ocolire.png")
new_image=preprocess_img(new_image)
image = np.expand_dims(new_image, axis=0)
pred = model.predict(image)
print(classification(str(np.argmax(pred)+1)))

new_image = cv2.imread("Examples/prioritate.png")
new_image=preprocess_img(new_image)
image = np.expand_dims(new_image, axis=0)
pred = model.predict(image)
print(classification(str(np.argmax(pred)+1)))

new_image = cv2.imread("Examples/depasire_camioane.jpg")
new_image=preprocess_img(new_image)
image = np.expand_dims(new_image, axis=0)
pred = model.predict(image)
print(classification(str(np.argmax(pred)+1)))

new_image = cv2.imread("Examples/depasire.png")
new_image=preprocess_img(new_image)
image = np.expand_dims(new_image, axis=0)
pred = model.predict(image)
print(classification(str(np.argmax(pred)+1)))

new_image = cv2.imread("Examples/curba1.png")
new_image=preprocess_img(new_image)
image = np.expand_dims(new_image, axis=0)
pred = model.predict(image)
print(classification(str(np.argmax(pred)+1)))

new_image = cv2.imread("Examples/curba2.png")
new_image=preprocess_img(new_image)
image = np.expand_dims(new_image, axis=0)
pred = model.predict(image)
print(classification(str(np.argmax(pred)+1)))