import pandas as pd
import numpy as np

from database import Database
from datetime import datetime
from tqdm import tqdm


class Importer:

    def __init__(self):
        self.database = Database()

    def import_syn_table(self, dataframe, collection_name):
        """
        Imports in MongoDB a synapse entity (table).

        Args:
            dataframe: The synapse table in a DataFrame format
            collection_name: The collection name of MongoDB to be saved
        """

        # Rename field "data.csv" to "data_id" since MongoDB has issues with dot in keys.
        dataframe = dataframe.rename(columns={"data.csv": "data_id"})

        docs = dataframe.to_dict(orient='records')

        collection = self.database.get_or_create_collection(collection_name)
        collection.insert_many(docs)

    def import_embedded_data(self, dataframe, collection_name):
        """
        Imports from the file handle ids all the embedded files.

        Args:
            dataframe: The file handle ids in DataFrame format
            collection_name: The collection name of MongoDB to be saved

        Returns:
            A list with tuples that weren't successfully imported into database.
            Format is:
                (data_id, reason)
        """

        collection = self.database.get_or_create_collection(collection_name)

        failed_ids = []
        for index, row in dataframe.iterrows():
            try:
                print("Importing data id: {}".format(index))

                data_id = index
                filepath = row.values[0]

                df_inner = pd.read_csv(filepath)

                # Also save the data_id to know in which record it refers in order to join them later
                df_inner['data_id'] = data_id

                # There are files with no records
                if not df_inner.empty:
                    docs = df_inner.to_dict(orient='records')
                else:
                    # Empty file so create an empty dict and fill it with None
                    new_doc = {}
                    docs = df_inner.to_dict(orient='dict')
                    for k, v in docs.items():
                        new_doc[k] = None

                self.database.insert_to_collection(collection, docs)
            except Exception as e:
                failed_ids.append((index, str(e)))
        return failed_ids

    def merge_original_and_embedded_data(self, new_coll_name, activity_to_keep='HKQuantityTypeIdentifierStepCount'):
        new_coll = self.database.get_or_create_collection(new_coll_name)

        hk_data = pd.read_csv('../../data/healthkit_data.csv', index_col=0)
        hk_data = hk_data.reset_index(drop=True)
        file_ids = pd.read_csv('../../data/file_handles_healthkit_data.csv',
                               header=0,
                               names=['data_id', 'cache_filepath'])

        # Get all users that we have in the Healthkit data dataset
        all_users = hk_data['healthCode'].unique()

        for user in tqdm(all_users):

            # For the current user find all his data ids that have embedded records.
            # We join based on data_id in the file handles df.
            user_df = hk_data[hk_data['healthCode'] == user]
            df_merged = user_df.merge(right=file_ids, how='inner')

            # 'createOn' field is on Unix epoch so convert it to datetime
            df_merged['createdOn'] = df_merged['createdOn'].apply(lambda x: datetime.fromtimestamp(x / 1000))

            for index, row in df_merged.iterrows():
                cache_file = row['cache_filepath']

                try:
                    embedded_df = pd.read_csv(cache_file)
                except UnicodeDecodeError as e:
                    # There are two "corrupted" files from the whole Healthkit data  that
                    # need to be opened with Unicode encoding instead of utf-8
                    print("Opening file with unicode encoding: {} ".format(cache_file))
                    embedded_df = pd.read_csv(cache_file, encoding='unicode_escape')
                except Exception as e:
                    print("Exception raised. Corrupted file:", str(e))
                    continue

                # Keep only steps count activity
                embedded_df = embedded_df[embedded_df['type'] == activity_to_keep]

                # Convert to datetime objects to UTC only
                embedded_df['startTime'] = pd.to_datetime(embedded_df['startTime'], errors='coerce', utc=True)
                embedded_df['endTime'] = pd.to_datetime(embedded_df['endTime'], errors='coerce', utc=True)

                # Replace NaT with None for MongoDB
                embedded_df.replace({pd.NaT: None}, inplace=True)

                # Check if current cache file contains any steps count activity
                if embedded_df.empty:
                    continue
                else:
                    # Populate with user's info the data and insert
                    for index, value in row.iteritems():
                        embedded_df[index] = value

                    # Drop unnecessary columns before insert to mongo
                    embedded_df.drop(['data_id', 'cache_filepath'], axis=1, inplace=True)

                    final_docs = embedded_df.to_dict(orient='records')
                    self.database.insert_to_collection(new_coll, final_docs)


if __name__ == '__main__':
    importer = Importer()
    importer.merge_original_and_embedded_data('healthkit_stepscount_singles')
