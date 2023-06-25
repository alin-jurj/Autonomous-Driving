import pymongo
from pymongo import MongoClient
conn_str = "mongodb+srv://alinjurj00:parola123@cozmoapp.frpgwua.mongodb.net/?retryWrites=true&w=majority"
try:
    client = pymongo.MongoClient(conn_str)
except Exception:
    print("Error:" + Exception)

myDb = client["test"]

collection = myDb["requests"]
def db_location():


    results = collection.find()
    if results:
        for result in results:
            ROAD_name = result['destination']
            if ROAD_name == "Church":
                ROAD = 2
            if ROAD_name == 'School':
                ROAD = 3
            if ROAD_name == 'Police':
                ROAD = 1

            return ROAD
    else:
        return 0

def delete_request():
    collection.delete_many({})
