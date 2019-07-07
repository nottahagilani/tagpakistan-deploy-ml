import flask
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from more_itertools import sort_together
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
import re
import tensorflow
import requests


# EXIF Helper functions 

def get_if_exist(data, key):
    if key in data:
        return data[key]
    return None


def convertStandard(value):
    d0 = value[0][0]
    d1 = value[0][1]
    d = float(d0) / float(d1)

    m0 = value[1][0]
    m1 = value[1][1]
    m = float(m0) / float(m1)

    s0 = value[2][0]
    s1 = value[2][1]
    s = float(s0) / float(s1)

    return d + (m / 60.0) + (s / 3600.0)


def getExifData(image):  # pass image path --> turns image to a php-ish standard
    exif_data = {}
    img = image
    dataExif = img._getexif()

    if dataExif:
        for tag, value in dataExif.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gpsData = {}
                for each in value:
                    subDecoded = GPSTAGS.get(each, each)
                    gpsData[subDecoded] = value[each]

                exif_data[decoded] = gpsData
            else:
                exif_data[decoded] = value
    return exif_data


def exif_extract_information(image):  # takes a exif Data Dictionary
    latitude = None
    longitude = None
    date = None
    time = None
    exif_data = getExifData(image)

    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]
        gps_latitude = get_if_exist(gps_info, "GPSLatitude")
        gps_latitude_ref = get_if_exist(gps_info, 'GPSLatitudeRef')
        gps_longitude = get_if_exist(gps_info, 'GPSLongitude')
        gps_longitude_ref = get_if_exist(gps_info, 'GPSLongitudeRef')
        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            latitude = convertStandard(gps_latitude)
            if gps_latitude_ref != "N":
                latitude = 0 - latitude
            longitude = convertStandard(gps_longitude)
            if gps_longitude_ref != "E":
                longitude = 0 - longitude

    if "DateTimeDigitized" in exif_data:
        timeTaken = str(exif_data["DateTimeDigitized"])
        time_arr = timeTaken.split(' ')
        date = time_arr[0]
        time = time_arr[1]
        date_split = date.split(':')
        date = str(date_split[2]) + '/' + \
            str(date_split[1]) + '/' + str(date_split[0])

    else:
        timeTaken = 0

    return latitude, longitude, date, time

# global model_places, graph_places, places_labels
base_model = InceptionV3(weights=None, include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(162, activation='softmax')(x)
model_places = Model(inputs=base_model.input, outputs=predictions)
model_places.load_weights('metadata/places_weights.hdf5')
graph_places = tensorflow.get_default_graph()
print("Model loaded..")

places_labels = {}
fd = open('metadata/places_categories.txt', 'r')
for x in fd.readlines():
    y = x.split(' ')
    cat = y[0]
    indx = y[1].strip()
    places_labels[int(indx)] = cat
fd.close()
print("Categories lables loaded..")

del (base_model)
del (predictions)
del (x)


app = flask.Flask(__name__)


def prepare_image(image, target_dim):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_dim)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def image_pass_places(input_img):

    global model_places, graph_places, places_labels

    (latitude, longitude, date, time) = exif_extract_information(input_img)
    img_target_size = (224, 224)
    img = prepare_image(input_img, img_target_size)

    with graph_places.as_default():
        features = model_places.predict(img)

    cat_list = []
    prob_list = []

    for x in range(0, 162):
        cat_list.append(places_labels[x])
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


@app.route('/', methods=["POST"])
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


if __name__ == '__main__':
    app.run()
