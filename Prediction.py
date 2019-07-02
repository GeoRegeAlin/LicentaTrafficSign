from keras.models import load_model
from keras.preprocessing import image
from skimage import color, transform
import numpy as np
import json

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
new_image = image.load_img("Examples/Limita20km.jfif")
new_image=preprocess_img(new_image)
image = np.expand_dims(new_image, axis=0)
pred = model.predict(image)
print(classification(str(np.argmax(pred)+1)))