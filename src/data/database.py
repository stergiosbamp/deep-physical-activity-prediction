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


class HealthKitDatabase(Database):
    HEALTHKIT_STEPSCOUNT_COLLECTION = 'healthkit_stepscount_singles'

    def __init__(self):
        super().__init__()
        self.healthkit_collection = self.database.get_collection(self.HEALTHKIT_STEPSCOUNT_COLLECTION)

    def get_all_healthkit_users(self):
        cursor = self.healthkit_collection.distinct('healthCode')
        return cursor

    def set_healthkit_index(self, field):
        self.healthkit_collection.create_index(field)

    def get_records_by_user(self, user_code):
        cursor = self.healthkit_collection.find({"healthCode": user_code})
        return cursor
