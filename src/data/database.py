from pymongo import MongoClient
from pymongo.database import Collection


class Database:

    DATABASE_NAME = 'deep_physical_activity_prediction_db'

    def __init__(self):
        self.client = MongoClient('localhost', 27017)
        self.database = self.client.get_database(self.DATABASE_NAME)

    def get_or_create_collection(self, collection):
        return self.database.get_collection(collection)

    @staticmethod
    def insert_to_collection(collection: Collection, docs):
        if type(docs) is list:
            collection.insert_many(docs)
        else:
            collection.insert_one(docs)
