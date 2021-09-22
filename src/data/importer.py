import pandas as pd

from database import Database
from dateutil import parser
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

    def merge_original_and_embedded(self, original_coll_name, embedded_coll_name, new_coll_name):
        original_coll = self.database.get_or_create_collection(original_coll_name)
        embedded_coll = self.database.get_or_create_collection(embedded_coll_name)
        new_coll = self.database.get_or_create_collection(new_coll_name)

        # Fetch all distinct users we have in the dataset
        users = original_coll.distinct('healthCode')

        for user in tqdm(users):
            print("Processing user:", user)
            user_docs = original_coll.find({'healthCode': user})
            user_data_ids = []

            final_doc = {}
            i = 0
            # Find all data_ids for the current user
            for user_doc in user_docs:
                # Do this only one time to get the user's elements
                if i == 0:
                    # 'createOn' field is on Unix epoch
                    user_doc['createdOn'] = datetime.fromtimestamp(user_doc['createdOn']/1000)
                    # Copy to final doc the original doc
                    final_doc = user_doc.copy()
                    i += 1
                data_id = user_doc['data_id']
                user_data_ids.append(data_id)

            # Find all docs that have any of those user data ids. Some do not belong in the stepsCount activity.
            # So we filter those with the $in operator of mongo.
            # We could instead iterate those user data ids and query them one by one, but it's really
            # time-consuming. Mongo handles that much better.

            data_id_docs_cursor = embedded_coll.find({"data_id": {"$in": user_data_ids}})

            for data_id_doc in data_id_docs_cursor:
                data_id_doc['startTime'] = parser.parse(data_id_doc['startTime'])
                data_id_doc['endTime'] = parser.parse(data_id_doc['endTime'])

                # Copy to final doc the embedded doc
                for k, v in data_id_doc.items():
                    final_doc[k] = v

                # drop data_id field
                final_doc.pop('data_id')
                self.database.insert_to_collection(new_coll, final_doc)

    def keep_stepcount_data_only(self):
        coll = self.database.get_or_create_collection("healthkit_data_embedded")
        step_count_records = coll.find({"type": "HKQuantityTypeIdentifierStepCount"})

        step_collection = self.database.get_or_create_collection("HK_embedded_stepcount")

        for record in step_count_records:
            self.database.insert_to_collection(step_collection, record)


if __name__ == '__main__':
    importer = Importer()
    # Significantly reduces the relevant documents that we need before merging
    # importer.keep_stepcount_data_only()
    importer.merge_original_and_embedded(original_coll_name='healthkit_data',
                                         embedded_coll_name='HK_embedded_stepcount',
                                         new_coll_name='hk_stepscount_singles')
