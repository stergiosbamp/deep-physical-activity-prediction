import os

from pymongo import MongoClient
from pymongo.database import Collection


class Database:
    """
    Convenient class to interact with the MongoDB

    Attributes:
        client (pymongo.MongoClient): The client of the pymongo.
        database (pymongo.database.Database): The database of the project.
    """

    DATABASE_NAME = 'deep_physical_activity_prediction_db'

    def __init__(self):
        host = os.getenv("MONGODB_HOST")
        port = os.getenv("MONGODB_PORT")
        self.client = MongoClient(host, int(port))
        self.database = self.client.get_database(self.DATABASE_NAME)

    def get_or_create_collection(self, collection):
        """
        Function to create or get a collection to perform various database operations.

        Args:
            collection (str): The collection name to get or to create.

        Returns:
            (pymongo.database.Collection): The mongo collection.
        """

        return self.database.get_collection(collection)

    @staticmethod
    def insert_to_collection(collection: Collection, docs):
        """
        Function to insert one or many documents to a MongoDB collection.

        Args:
            collection (pymongo.database.Collection): The collection to insert the document(s).
            docs (Union[list, dict]: Either a list with docs or a single document as a dict.
        """

        if type(docs) is list:
            collection.insert_many(docs)
        else:
            collection.insert_one(docs)


class HealthKitDatabase(Database):
    """
    A more detailed class which inherits from "Database" class, for performing useful and common operations for the
    HealthKit data.

    Attributes:
        All of the parent class plus:

        healthkit_collection (pymongo.database.Collection): The main collection of the project named
            "healthkit_stepscount_singles".

    """

    HEALTHKIT_STEPSCOUNT_COLLECTION = 'healthkit_stepscount_singles'

    def __init__(self, port):
        super().__init__(port)
        self.healthkit_collection = self.database.get_collection(self.HEALTHKIT_STEPSCOUNT_COLLECTION)

    def get_all_healthkit_users(self):
        """
        Function to get all distinct users that exist in the collection.

        Returns:
            (pymongo.cursor.Cursor): The cursor that holds the results.
        """

        cursor = self.healthkit_collection.distinct('healthCode')
        return cursor

    def set_healthkit_index(self, field):
        """
        Function that sets an index on a specific field for fast retrieval of documents.
        Mainly used in the 'healthCode' which is used really often.

        Args:
            field (str): The field to set the index.
        """

        self.healthkit_collection.create_index(field)

    def get_records_by_user(self, user_code):
        """
        Function that returns all of the user's records (documents) from the collection.

        Args:
            user_code (str): The unique user code

        Returns:
            (pymongo.cursor.Cursor): The cursor that holds the results.
        """

        cursor = self.healthkit_collection.find({"healthCode": user_code})
        return cursor
