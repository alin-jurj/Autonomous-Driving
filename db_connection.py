import pymongo

conn_str = ""
try:
    client = pymongo.MongoClient(conn_str)
except Exception:
    print("Error:"+ Exception)

myDb = client