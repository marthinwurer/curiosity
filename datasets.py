import sqlite3
import subprocess
from sqlite3 import Cursor, Connection

from PIL import Image

"""
Image dataset:
images have integer ids and paths
"""

class Item(object):
    def __init__(self, path, _id=None):
        self._id = _id
        self.path = path

    def __repr__(self):
        return "Item(path=%r)" % (self.path,)

    def load(self):
        pass


def check_if_exists(cursor: Cursor, table_name):
    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name=?;""", (table_name,))

    result = len(cursor.fetchall())
    return result > 0


def build_image_table(conn: Connection):

    cursor = conn.cursor()

    if check_if_exists(cursor, "items"):
        return

    cursor.execute("""
        CREATE TABLE items (
            id integer primary key autoincrement ,
            path text not null )""")

    conn.commit()

def save_image_path(conn: Connection, path):

    cursor = conn.cursor()
    cursor.execute("""INSERT INTO items(path) values (?);""", (path,))
    row_id = cursor.lastrowid
    conn.commit()

    return Item(path, row_id)

def load_image(item: Item):
    return Image.open(item.path)

def load_path(conn: Connection, path_id):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, path FROM items
        WHERE id=?;""", (path_id,))
    return Item(cursor.fetchone()[1], path_id)


def display_image(image: Image):
    image.show()

def display_item(item: Item):
    image = load_image(item)
    display_image(image)




def main():

    subprocess.call(["rm", "mydatabase.db"])

    conn = sqlite3.connect("mydatabase.db")

    # create a table
    build_image_table(conn)

    path = "/mnt/nas/datasets/visualgenome/VG_100K/713947.jpg"

    item = save_image_path(conn, path)

    print(item)

    new_item = load_path(conn, 1)

    display_item(new_item)



if __name__ == "__main__":
    main()

