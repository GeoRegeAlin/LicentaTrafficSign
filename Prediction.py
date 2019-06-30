from keras.models import load_model
from keras.preprocessing import image
from skimage import color, transform
import numpy as np
import json
IMG_SIZE = 48

def preprocess_img(img):
    hsv = color.rgb2hsv(img)
    img = color.hsv2rgb(hsv)
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    img = np.rollaxis(img, -1)

    return img

def clasificare(id):
    with open('IdentitateSemn.txt') as json_file:
        data=json.load(json_file)
    print(data[str(id)])


model = load_model("CNNmodel.h5")
new_image = image.load_img("Examples/5.jfif")
new_image=preprocess_img(new_image)
image = np.expand_dims(new_image, axis=0)
pred = model.predict(image)
print(clasificare(str(np.argmax(pred)+1)))