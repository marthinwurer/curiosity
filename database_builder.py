import argparse
import glob
import os
import sqlite3
import tempfile
from sqlite3 import Connection
from urllib.parse import urlparse

import requests

from datasets import Item, save_image_path

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
                Item(path=full_path).save_with(conn)
                print(full_path)
            except sqlite3.IntegrityError as e:
                print("image already exists: %s" % (full_path,))


def store_item_urls(base_dir, conn: Connection):
    for root, dirs, files in os.walk(base_dir):
        print(root, dirs, files)
        images_dir = os.path.join(root, "IMAGES")
        # find images files
        for file in glob.iglob(root + "/urls.txt"):
            print("found url file: %s" % (file,))
            # subprocess.call(["wget", "-nc", "-q", "--timeout=5", "--tries=2", "-i", file, "-P", images_dir])

            with open(file) as f:
                for line in f:
                    line = line.strip()
                    try:
                        Item(url=line).save_with(conn)
                    except sqlite3.IntegrityError as e:
                        print("line already exists: %s" % (line,))


def download_item(conn: Connection, item: Item, dir):


    # download file
    filename = os.path.basename(urlparse(item.url).path)
    print("Filename: %s" % (filename,))
    file_path = os.path.join(dir, filename)

    r = requests.get(item.url, stream=True)
    print("Headers: %s" % (r.headers,))
    with open(file_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
    # hash file
    # check database to see if the hash exists
    # if it does, set this to an alias
    # compute path
    # move file
    # update database





def main():
    parser = argparse.ArgumentParser(description='Short sample app')

    parser.add_argument('--wipe', action="store_true")
    parser.add_argument('database')
    parser.add_argument('base_dir')

    args = parser.parse_args()

    print(args)

    database_path = args.database
    base_dir = args.base_dir
    conn = sqlite3.connect(database_path)
    # create a table
    if args.wipe:

        conn.close()  # close then reopen after it's wiped
        print("Wiping database at %s" % (database_path,))
        os.remove(database_path)
        conn = sqlite3.connect(database_path)
        Item.build_table(conn)

    # store_image_paths(args.base_dir, conn)
    store_item_urls(base_dir, conn)

    # download missing urls
    temp_dir = tempfile.mkdtemp()
    print("Done!")


if __name__ == "__main__":
    main()




