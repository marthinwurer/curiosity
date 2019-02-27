import argparse
import os
import sqlite3
from sqlite3 import Connection

from datasets import build_image_table, save_image_path

"""
$ python database_builder.py /mnt/nas/datasets/database.db /mnt/nas/datasets/visualgenome/VG_100K/
python database_builder.py --wipe /mnt/nas/datasets/database.db /mnt/nas/datasets/test_data/
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
            try:
                save_image_path(conn, full_path)
                print(full_path)
            except sqlite3.IntegrityError as e:
                print("image already exists: %s" % (full_path,))


def main():
    parser = argparse.ArgumentParser(description='Short sample app')

    parser.add_argument('--wipe', action="store_true")
    parser.add_argument('database')
    parser.add_argument('base_dir')

    args = parser.parse_args()

    print(args)

    database_path = args.database
    conn = sqlite3.connect(database_path)
    # create a table
    if args.wipe:

        conn.close()  # close then reopen after it's wiped
        print("Wiping database at %s" % (database_path,))
        os.remove(args.database)
        conn = sqlite3.connect(database_path)
        build_image_table(conn)

    store_image_paths(args.base_dir, conn)



if __name__ == "__main__":
    main()




