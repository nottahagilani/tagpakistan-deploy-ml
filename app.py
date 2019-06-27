import flask
from exifExtractor import exifExtractInformation
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import os.path
from more_itertools import sort_together
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import re

app = flask.Flask(__name__)

#global_variables
model_places = None
model_birds = None
places_cats = {}
birds_cats = {}

def load_model_places():
    # global model_places
    base_model = InceptionV3( weights=None , include_top=False )
    x = base_model.output
    x = GlobalAveragePooling2D()( x )
    x = Dense( 1024 , activation='relu' )( x )
    predictions = Dense( 162 , activation='softmax' )( x )
    global model_places
    model_places = Model( inputs=base_model.input , outputs=predictions )
    model_places.load_weights('metadata/places_weights.hdf5')

def load_model_birds():
    # global model_birds
    base_model = InceptionV3( weights=None , include_top=False )
    x = base_model.output
    x = GlobalAveragePooling2D()( x )
    x = Dense( 1024 , activation='relu' )( x )
    predictions = Dense( 200 , activation='softmax' )( x )
    global model_birds
    model_birds = Model( inputs=base_model.input , outputs=predictions )
    model_birds.load_weights( 'metadata/birds_weights.hdf5' )

def prepare_image(image, target_dim):

    if image.mode != "RGB":
        image = image.convert( "RGB" )

    image = image.resize(target_dim)
    image = img_to_array( image )
    image = np.expand_dims( image , axis=0 )
    image = preprocess_input( image )

    return image

def load_labels():

    # global places_cats
    # global birds_cats

    fd = open( 'metadata/places_categories.txt' , 'r' )
    # places_cats = {}
    for x in fd.readlines():
        y = x.split( ' ' )
        cat = y[0]
        indx = y[1].strip()
        places_cats[int( indx )] = cat
    fd.close()

    fd = open( 'metadata/birds_categories.txt' , 'r' )
    # birds_cats = {}
    for x in fd.readlines():
        y = x.split( ' ' )
        cat = y[0]
        indx = y[1].strip()
        birds_cats[int( indx )] = cat
    fd.close()

def image_pass_places(input_img):

    (latitude, longitude, date, time) = exifExtractInformation(input_img)
    img_target_size = (224,224)
    img = prepare_image(input_img, img_target_size)
    global model_places
    features = model_places.predict( img )

    cat_list = []
    prob_list = []

    for x in range( 0 , 162 ):
        cat_list.append( places_cats[x] )
        prob_list.append( features[0 , x] )

    sorted_list = sort_together( [prob_list , cat_list] , reverse=True )
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

def image_pass_birds(input_img):
    (latitude, longitude, date, time) = exifExtractInformation(input_img)
    img_target_size = (299,299)
    input_img = prepare_image(input_img, img_target_size)
    global model_birds
    features = model_birds.predict( input_img )

    cat_list = []
    prob_list = []

    for x in range( 0 , 200 ):
        cat_list.append( birds_cats[x] )
        prob_list.append( features[0 , x] )

    sorted_list = sort_together( [prob_list , cat_list] , reverse=True )
    prob_list = sorted_list[0]
    cat_list = sorted_list[1]

    top_5_cat = cat_list[0:5]

    jsonString = ''
    for x in top_5_cat:
        jsonString += jsonString + x + ' '


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
        if flask.request.files.get("image"):

            img = flask.request.files["image"].read()
            img = Image.open(io.BytesIO(img))

            info_dict = image_pass_places( img )

            return_response["tags"] = info_dict["tags"]
            return_response["latitude"] = info_dict["latitude"]
            return_response["longitude"] = info_dict["longitude"]
            return_response["date"] = info_dict["date"]
            return_response["time"] = info_dict["time"]

            return_response["success"] = 1

    return flask.jsonify(return_response)

@app.route('/predict_birds', methods=["POST"])
def get_response_birds():

    return_response = {"success": 0}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            img = flask.request.files["image"].read()
            img = Image.open(io.BytesIO(img))
            info_dict = image_pass_birds( img )

            return_response["tags"] = info_dict["tags"]
            return_response["latitude"] = info_dict["latitude"]
            return_response["longitude"] = info_dict["longitude"]
            return_response["date"] = info_dict["date"]
            return_response["time"] = info_dict["time"]

            return_response["success"] = 1

    return flask.jsonify(return_response)

@app.route('/')
def print_main():
    return '<h1>Welcome to TagPakistan ML API</h1>'
	
if __name__ == '__main__':
    print(" The server has started ..... \n");
    load_model_places()
    print(" The places model has been loaded ..... \n");
    load_model_birds()
    print( " The birds model has been loaded ..... \n" ) ;
    load_labels()
    print(" The Class labels have been loaded.... \n");
    print(" Sever ready for images ... \n");
    app.run(debug='True')

# print(" The server has started ..... \n");
# load_model_places()
# print(" The places model has been loaded ..... \n");
# load_model_birds()
# print( " The birds model has been loaded ..... \n" ) ;
# load_labels()
# print(" The Class labels have been loaded.... \n");
# print(" Sever ready for images ... \n");
# app.run()