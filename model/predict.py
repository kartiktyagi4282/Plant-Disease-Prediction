from keras.models import model_from_json
import cv2
import numpy as np
import json

# loading model from json
json_file = open('weights/VGG16/VGG16_model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("weights/VGG16/vgg16_best.hdf5")
#print("Loaded model from disk")

# file path
im = cv2.imread("test images/tomato_healthy.JPG")
im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (256, 256))
im = np.expand_dims(im, axis =0)

outcome = model.predict(im)

# loading labels file to convert to string
with open('labels.json', 'r') as fp:
    data = json.load(fp)

pred_disease=data[str(np.argmax(outcome))]
print("predicted output:",pred_disease)