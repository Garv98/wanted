import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../.env')

MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB = os.getenv('MONGODB_DB')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION')

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]
collection = db[MONGODB_COLLECTION]

with open('crime_dataset_india_cleaned.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

if isinstance(data, list):
    collection.insert_many(data)
else:
    collection.insert_one(data)

print("Data imported successfully to MongoDB Atlas.")
