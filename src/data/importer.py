import pandas as pd

from database import Database


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

    def import_joined_data(self):
        pass
