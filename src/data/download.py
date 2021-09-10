import pandas as pd
import synapseclient
import os

from dotenv import load_dotenv

from database import Database
from config import SYNAPSE_TABLES, TABLES_EMBEDDED_DATA_COLUMN


class Downloader:
    def __init__(self):
        self.syn = synapseclient.Synapse()
        self.database = Database()

    def login(self, user, password):
        self.syn.login(user, password)

    def save_synapse_table_in_database(self, table_name, entity_id):
        entity = self.syn.tableQuery("SELECT * FROM {}".format(entity_id))
        df = entity.asDataFrame()
        docs = df.to_dict(orient='records')

        collection = self.database.get_or_create_collection(table_name)
        collection.insert_many(docs)
        print("Downloaded and saved table: {} into MongoDB collection".format(table_name))


if __name__ == '__main__':
    load_dotenv()

    user = os.getenv('SYNAPSE_USER')
    password = os.getenv('SYNAPSE_PASSWORD')

    downloader = Downloader()

    downloader.login(user=user, password=password)

    for table, entity_id in SYNAPSE_TABLES.items():
        downloader.save_synapse_table_in_database(table, entity_id)
