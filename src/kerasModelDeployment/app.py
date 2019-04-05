from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from skimage.transform import rescale

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model('model/model_using_keras.h5')
model._make_predict_function()  
print('Model is loaded..start the browser  http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(32, 32))
    image_arr = image.img_to_array(img)
    image_arr = np.expand_dims(image_arr, axis=0)
    image_arr /= 255
    preds = model.predict(image_arr)
    prediction_result = decodePredictions(preds)
    return prediction_result

def decodePredictions(pred_array):
    answer = np.argmax(pred_array)
    result = ""
    if answer == 0:
        result="c0: safe driving"
    elif answer == 1:
        result="c1: texting - right"
    elif answer == 2:
        result="c2: talking on the phone - right"
    elif answer == 3:
        result="c3: texting - left"
    elif answer == 4:
        result="c4: talking on the phone - left"
    elif answer == 5:
        result="c5: operating the radio"
    elif answer == 6:
        result="c6: drinking"
    elif answer == 7:
        result="c7: reaching behind"
    elif answer == 8:
        result="c8: hair and makeup"
    elif answer == 9:
        result="c9: talking to passenger"
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        cuur_dir_path = os.path.dirname(__file__)
        curr_file_path = os.path.join(cuur_dir_path, 'server_data', secure_filename(uploaded_file.filename))
        uploaded_file.save(curr_file_path)

        predsiction_result= model_predict(curr_file_path, model)
        return predsiction_result
    return None

if __name__ == '__main__':
     http_server = WSGIServer(('', 5000), app)
     http_server.serve_forever()
