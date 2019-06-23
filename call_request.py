import requests
import os
rest_api_call = "http://localhost:5000/predict_places"
image_path = "images/"
list_dir = os.listdir("images")

for x in list_dir:
    img = open(image_path+x,"rb").read()
    payload = {"image":img}

    r = requests.post(rest_api_call,files = payload).json()

    if r["success"] == 1:
        print("File Name: " + x)
        print( "tags: " + str(r["tags"]) )
        print( "latitude: " + str(r["latitude"]) )
        print( "longitude: " + str(r["longitude"]) )
        print( "data: " + str(r["date"]) )
        print( "time: " + str(r["time"]) )
        print('\n')
    else:
        print("Request Failed")

