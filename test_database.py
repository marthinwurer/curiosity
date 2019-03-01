import sqlite3
import unittest

from database_builder import download_item
from datasets import Item


class DatasetTest(unittest.TestCase):
    def test_image_download(self):
        url = "https://homepages.cae.wisc.edu/~ece533/images/cat.png"
        path = "./data"
        database = "./data/database.db"
        conn = sqlite3.connect(database)
        item = download_item(conn, Item(url=url), path)





if __name__ == '__main__':
    unittest.main()
