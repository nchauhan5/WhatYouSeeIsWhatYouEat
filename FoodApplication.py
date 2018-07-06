from network import VGG16
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import backend as k
import numpy as np

from flask import Flask, render_template, request
import os

import requests
import json
import urllib
import string

FOOD_FOLDER = "D:\\Projects\\Python\\FlaskTutorial\\static\\food\\"
TRAINING_DATA_FOLDER = "D:\\Projects\\Python\\FineTunedImageRecognition\\training_data\\"
RELATIVE_FOOD_FOLDER = os.path.join('static', 'food')
FOOD_PROBABILITY_THRESHOLD = 0.30

api_key = 'xyL0hiueReFGrbI3yR3Y36I2ZMsfFl8caH3foP3u'

app = Flask(__name__)

app.config['FOOD_FOLDER'] = FOOD_FOLDER
app.config['RELATIVE_FOOD_FOLDER'] = RELATIVE_FOOD_FOLDER
app.config['TRAINING_DATA_FOLDER'] = TRAINING_DATA_FOLDER
app.config['FOOD_PROBABILITY_THRESHOLD'] = FOOD_PROBABILITY_THRESHOLD


@app.route("/")
@app.route("/index")
def hello():
    return render_template(
        'index.html')


@app.route("/snapshot")
def snapshot():
    file_url = "http://192.168.10.1/media/?action=snapshot"
    s = requests.Session()
    s.auth = ('admin', '')
    r = s.get(file_url, stream=True)

    with open(os.path.join(app.config['FOOD_FOLDER'], 'test_image.jpg'), "wb") as food:
        for chunk in r.iter_content(chunk_size=1024):

            # writing one chunk at a time to pdf file
            if chunk:
                food.write(chunk)

    full_filename = os.path.join(app.config['RELATIVE_FOOD_FOLDER'], 'test_image.jpg')
    return render_template('snapshot.html', food_snapshot=full_filename)


@app.route('/viewResults', methods=['POST', 'GET'])
def view_results():
    base_model = VGG16.VGG16(include_top=False, weights=None)
    x = base_model.output
    x = Dense(128)(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(101, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights("cv-tricks_fine_tuned_model.h5")

    input_shape = (224, 224)  # Assumes 3 channel image
    image = load_img(os.path.join(app.config['FOOD_FOLDER'], 'test_image.jpg'), target_size=input_shape)
    image = img_to_array(image)  # shape is (224,224,3)
    image = np.expand_dims(image, axis=0)  # Now shape is (1,224,224,3)

    image = image / 255.0

    preds = model.predict(image)

    food_prob_array = preds[0]
    food_prob_list = food_prob_array.tolist()
    k.clear_session()

    owd = os.getcwd()
    os.chdir('D:\\Projects\\Python\\FineTunedImageRecognition\\training_data')
    all_subdir = [d for d in os.listdir('.') if os.path.isdir(d)]
    os.chdir(owd)

    food_dict = dict(zip(all_subdir, food_prob_list))

    sorted_d = sorted(food_dict.items(), key=lambda food_dict: food_dict[1], reverse=True)
    print(sorted_d)
    dish_name = string.capwords(str(sorted_d[0][0]).replace("_", " "))
    prob_dish = sorted_d[0][1]

    if prob_dish < app.config['FOOD_PROBABILITY_THRESHOLD']:
        error_message = "The image was not identified as any Dish. Please capture it again by clicking 'Take A Snapshot Again' button"
        return render_template('noResults.html', error_message=error_message)

    else:
        print("The dish was identified to be {} with {} probability".format(dish_name, prob_dish))
        nutrients, error = get_nutrients_data_from_usda(dish_name)
        if nutrients is not None:
            weight = request.args.get('weight')
            print("*********** " + weight + " ****************")
            if not weight:
                weight = 100
            weight_multiplier = float(weight) / 100
            print(nutrients)
            for nutrient in nutrients:
                nutrient['value'] = str(round(float(nutrient['value']) * weight_multiplier, 2))
        if error is not None:
            print(error)

        could_also_be_dish_name = sorted_d[1][0]
        could_also_be_dish_name_display_name = string.capwords(could_also_be_dish_name.replace("_", " "))
        #   print("The dish could also be {} or {} ".format(sorted_d[1][0], sorted_d[2][0]))
        return render_template('viewResults.html', dish_name=dish_name, nutrients=nutrients, display_second=could_also_be_dish_name_display_name, second_dish=could_also_be_dish_name, error=error, weight=weight)


@app.route("/viewResults/<string:name>/", methods=['POST', 'GET'])
def get_second_dish(name):
    dish_name = string.capwords(name.replace("_", " "))
    nutrients, error = get_nutrients_data_from_usda(dish_name)
    if nutrients is not None:
        weight = request.args.get('weight')
        if not weight:
            weight = 100
        weight_multiplier = float(weight) / 100
        print(nutrients)
        for nutrient in nutrients:
            nutrient['value'] = str(round(float(nutrient['value']) * weight_multiplier, 2))
    if error is not None:
        print(error)
    return render_template('second_probable_dish.html', dish_name=dish_name, nutrients=nutrients, error=error, weight=weight)


# Clear browser cache after every request
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response


def get_ndbno(food_name):
    ndb_no = None
    resp_err = None
    url = 'https://api.nal.usda.gov/ndb/search/?format=json&q={}&sort=n&max=25&offset=0&api_key=xyL0hiueReFGrbI3yR3Y36I2ZMsfFl8caH3foP3u'.format(
        urllib.parse.quote(food_name))
    print(url)
    response = requests.get(url)
    resp_text = json.loads(response.text)

    # print(response.status_code)

    if 400 <= response.status_code < 500:
        resp_err = "A client-side error occurred. Error {}: Could not fetch data for {} from API".format(response.status_code, food_name)
        print(resp_err)
        return ndb_no, resp_err

    elif 500 <= response.status_code < 600:
        resp_err = 'A server-side error occurred. Error {}: Could not fetch data for {} from API'.format(response.status_code, food_name)
        print(resp_err)
        return ndb_no, resp_err

    for key in resp_text:
        if key == 'errors':
            resp_err = "The USDA database has no information about the identified dish i.e. {}".format(food_name)
            print(resp_err)
            return ndb_no, resp_err

    ndb_no = resp_text['list']['item'][0]['ndbno']
    return ndb_no, resp_err


def get_food_nutrients(ndb_number):
    nutrients = None
    resp_err = None
    url = 'https://api.nal.usda.gov/ndb/V2/reports?ndbno={}&type=b&format=json&api_key=xyL0hiueReFGrbI3yR3Y36I2ZMsfFl8caH3foP3u'.format(
            ndb_number)
    print(url)
    response = requests.get(url)

    if 400 <= response.status_code < 500:
        resp_err = "A client-side error occurred. Error {}: Could not fetch data for {} from API".format(response.status_code, ndb_number)
        print(resp_err)
        return nutrients, resp_err

    elif 500 <= response.status_code < 600:
        resp_err = 'A server-side error occurred. Error {}: Could not fetch data for {} from API'.format(response.status_code, ndb_number)
        print(resp_err)
        return nutrients, resp_err

    resp_text = json.loads(response.text)
    for key in resp_text['foods'][0]:
        if key == "error":
            resp_err = "The USDA database has no information about the identified dish number i.e. {}".format(ndb_number)
            print(resp_err)
            return nutrients, resp_err

    nutrients = resp_text['foods'][0]['food']['nutrients']
    return nutrients, resp_err


def print_details(nutrients, ndb_no):
    print('The nutrients for ndbno: ' + str(ndb_no) + ' are-' + '\n')

    i = 0
    while i < len(nutrients):
        print(str(nutrients[i]['name']) + ' : ' + str(nutrients[i]['value'] + ' ' + str(nutrients[i]['unit'])))
        i += 1


def get_nutrients_data_from_usda(food_name):
    nutrients = None
    ndb_no, error = get_ndbno(food_name)
    if ndb_no is not None:
        nutrients, error = get_food_nutrients(ndb_no)
        if nutrients is not None:
            print_details(nutrients, ndb_no)
    return nutrients, error


# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == "__main__":
    app.run()
