import flask
from exif_extractor import exif_extract_information
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from more_itertools import sort_together
import numpy as np
from PIL import Image
import io
import re
import tensorflow
import requests

app = flask.Flask(__name__)

# global instantiations
global_graph = tensorflow.get_default_graph()
places_cats = {}

# loading labels
fd = open('metadata/places_categories.txt', 'r')
for x in fd.readlines():
    y = x.split(' ')
    cat = y[0]
    indx = y[1].strip()
    places_cats[int(indx)] = cat
fd.close()

# loading model
base_model = InceptionV3(weights=None, include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(162, activation='softmax')(x)
model_places = Model(inputs=base_model.input, outputs=predictions)
model_places.load_weights('metadata/places_weights.hdf5')


def prepare_image(image, target_dim):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_dim)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def image_pass_places(input_img):
    (latitude, longitude, date, time) = exif_extract_information(input_img)
    img_target_size = (224, 224)
    img = prepare_image(input_img, img_target_size)

    with global_graph.as_default():
        features = model_places.predict(img)

    cat_list = []
    prob_list = []

    for x in range(0, 162):
        cat_list.append(places_cats[x])
        prob_list.append(features[0, x])

    sorted_list = sort_together([prob_list, cat_list], reverse=True)
    prob_list = sorted_list[0]
    cat_list = sorted_list[1]

    top_5_cat = cat_list[0:5]

    jsonString = []

    for x in top_5_cat:
        x = re.sub(r'[^a-zA-Z ]+', ' ', x)
        x = x.title()
        jsonString.append(x)

    return_dict = \
        {
            "tags": jsonString,
            "latitude": latitude,
            "longitude": longitude,
            "date": date,
            "time": time,
        }
    return return_dict

@app.route('/predict_places', methods=["POST"])
def get_response_places():
    return_response = {"success": 0}

    if flask.request.method == "POST":
        if flask.request.files.get("image_bytes"):
            img = flask.request.files["image_bytes"].read()
            img = Image.open(io.BytesIO(img))

            info_dict = image_pass_places(img)

            return_response["tags"] = info_dict["tags"]
            return_response["latitude"] = info_dict["latitude"]
            return_response["longitude"] = info_dict["longitude"]
            return_response["date"] = info_dict["date"]
            return_response["time"] = info_dict["time"]

            return_response["success"] = 1

        if flask.request.files.get("image_url"):
            img_url = flask.request.files["image_url"].read()
            img_request = requests.get(img_url.decode('UTF-8'))
            img = Image.open(io.BytesIO(img_request.content))

            info_dict = image_pass_places(img)

            return_response["tags"] = info_dict["tags"]
            return_response["latitude"] = info_dict["latitude"]
            return_response["longitude"] = info_dict["longitude"]
            return_response["date"] = info_dict["date"]
            return_response["time"] = info_dict["time"]

            return_response["success"] = 1

    return flask.jsonify(return_response)


@app.route('/')
def print_main():
    return '<h1>Welcome to TagPakistan ML API for Places2</h1>'


if __name__ == '__main__':
    app.run()
