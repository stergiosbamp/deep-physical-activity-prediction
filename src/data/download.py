import pandas as pd
import synapseclient
import pickle
import os

from dotenv import load_dotenv

from config import SYNAPSE_TABLES, TABLES_EMBEDDED_DATA_COLUMN
from importer import Importer


class Downloader:
    """
    Class that downloads data from Synapse. Makes use of their python client.

    Attributes:
        syn (synapseclient.Synapse): The Synapse instance.
    """

    def __init__(self, user, password):
        self.syn = synapseclient.Synapse()
        self.syn.login(user, password)

    def download_synapse_table_as_df(self, entity_id):
        """
        Downloads a whole Synapse table as a DataFrame.

        Args:
            entity_id (str): The synapse entity id as str.

        Returns:
            (pd.DataFrame): The table's data as a DataFrame.
        """

        entity = self.syn.tableQuery("SELECT * FROM {}".format(entity_id))
        df = entity.asDataFrame()

        # Rename 'data.csv' to 'data_id' for MongoDB reasons since it can't handle
        # keys with . (dot)
        df = df.rename(columns={"data.csv": "data_id"})
        return df

    def download_embedded_data_files(self, embedded_column, entity_id):
        """
        Downloads all embedded files for a Synapse table to default directory.
        This is under '/home/{USER}/.synapseCache/'.

        Note that the client is easy to look for already
        downloaded files to not download them again, if exist.

        Args:
            embedded_column (str): The name of the column that holds embedded files.
            entity_id (str): The synapse entity id as str.

        Returns:
            (dict): The dictionary that holds the id of the file with it's corresponding path where it's downloaded.
        """

        query_results = self.syn.tableQuery("SELECT * FROM {}".format(entity_id))
        embedded_data_file_ids = self.syn.downloadTableColumns(query_results, [embedded_column])
        return embedded_data_file_ids


if __name__ == '__main__':
    load_dotenv()

    user = os.getenv('SYNAPSE_USER')
    password = os.getenv('SYNAPSE_PASSWORD')

    downloader = Downloader(user=user, password=password)
    importer = Importer()

    # Download original tables
    for table, entity_id in SYNAPSE_TABLES.items():
        df = downloader.download_synapse_table_as_df(entity_id)

        # Save it as csv
        df.to_csv("../../data/{}.csv".format(table))

        # And import them to Mongo
        importer.import_syn_table(dataframe=df, collection_name=table)

    # Download embedded data
    for table_name, column in TABLES_EMBEDDED_DATA_COLUMN.items():
        entity_id = SYNAPSE_TABLES[table_name]

        file_handles = downloader.download_embedded_data_files(embedded_column=column, entity_id=entity_id)

        # Save the file handle ids as csv
        df_files = pd.DataFrame.from_dict(file_handles, orient='index')
        df_files.to_csv('../../data/file_handles_{}.csv'.format(table_name))

        # And import each file from the handle ids file with '_embedded' collection name
        failed_ids = importer.import_embedded_data(dataframe=df_files, collection_name=table_name + "_embedded")

        # Write in a pickle how many files are corrupted from all the embedded csv
        # Answer: Only 2 embedded files
        with open("../../failed_ids_{}.pkl".format(table_name), 'wb') as f:
            pickle.dump(failed_ids, f)
