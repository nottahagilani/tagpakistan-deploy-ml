from PIL import Image
from PIL.ExifTags import TAGS,GPSTAGS

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

def getExifData(image): #pass image path --> turns image to a php-ish standard
    exif_data = {}
    img = image
    dataExif = img._getexif()

    if dataExif:
        for tag, value in dataExif.items():
            decoded = TAGS.get(tag,tag)
            if decoded == "GPSInfo":
                gpsData = {}
                for each in value:
                    subDecoded = GPSTAGS.get(each, each)
                    gpsData[subDecoded] = value[each]

                exif_data[decoded] = gpsData
            else:
                exif_data[decoded] = value
    return exif_data


def exif_extract_information(image): #takes a exif Data Dictionary
    latitude = None
    longitude = None
    date = None;
    time = None;
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
        date = time_arr[0];
        time = time_arr[1];
        date_split = date.split(':')
        date = str(date_split[2]) + '/' + str(date_split[1]) + '/' + str(date_split[0]);

    else:
        timeTaken = 0

    return latitude, longitude, date, time







