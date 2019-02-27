import argparse
import os
import sqlite3
from sqlite3 import Connection

from datasets import build_image_table, save_image_path

"""
$ python database_builder.py /mnt/nas/datasets/database.db /mnt/nas/datasets/visualgenome/VG_100K/
"""

def store_image_paths(base_dir, conn: Connection):
    types = [".jpg", ".png", ".gif"]
    for root, dirs, files in os.walk(base_dir):
        print(files)
        basenames = [os.path.splitext(file) for file in files]
        print(basenames)
        image_files = filter(lambda x: os.path.splitext(x)[1] in types, files)
        for file in image_files:
            full_path = os.path.join(root, file)
            save_image_path(conn, full_path)
            print(full_path)



def main():
    parser = argparse.ArgumentParser(description='Short sample app')

    parser.add_argument('database')
    parser.add_argument('base_dir')

    args = parser.parse_args()

    print(args)

    conn = sqlite3.connect(args.database)
    # create a table
    build_image_table(conn)

    store_image_paths(args.base_dir, conn)



if __name__ == "__main__":
    main()




