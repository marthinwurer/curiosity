import sqlite3
import subprocess
from sqlite3 import Cursor, Connection

from PIL import Image

"""
Image dataset:
images have integer ids and paths
"""


class Item(object):

    @staticmethod
    def build_table(conn: Connection):
        cursor = conn.cursor()
        if check_if_table_exists(cursor, "items"):
            return
        cursor.execute("""
            CREATE TABLE items (
                id integer primary key autoincrement ,
                path text unique,
                url text unique,
                hash text unique,
                author text,
                title text,
                description text,
                alias integer
                )""")
        conn.commit()

    @staticmethod
    def load_id(conn: Connection, path_id):
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT path, url, hash, author, title, description, alias FROM items
            WHERE id=?;""", (path_id,))
        return Item(_id=path_id, **cursor.fetchone())

    def __init__(self, path=None, url=None, hash=None, author=None, alias=None,
                 description=None, title=None, _id=None):
        self._id = _id
        self.path = path
        self.url = url
        self.hash = hash
        self.author = author
        self.title = title
        self.description = description
        self.alias = alias  # if this aliases to another id

    def __repr__(self):
        return "Item(path=%r,url=%s,hash=%s,author=%s)" % (self.path, self.url, self.hash, self.author)

    def load_image(self):
        """Load the file into a PIL image"""
        return Image.open(self.path)

    def hash_file(self):
        """load the file and hash it"""
        pass

    def save_with(self, conn: Connection):
        if self._id is None:
            # insert it!
            cursor = conn.cursor()
            cursor.execute("""INSERT INTO items(path, url, hash, author) values (?,?,?,?);""",
                           (self.path, self.url, self.hash, self.author))
            self._id = cursor.lastrowid
            conn.commit()
        else:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE items
                SET
                    path = ?,
                    url = ?,
                    hash = ?,
                    author = ?,
                    title = ?,
                    description = ?,
                    alias = ?
                WHERE id = ?;""", (self.path, self.url, self.hash, self.author,
                                   self.title, self.description, self.alias, self._id))
            self._id = cursor.lastrowid
            conn.commit()


def check_if_table_exists(cursor: Cursor, table_name):
    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name=?;""", (table_name,))

    result = len(cursor.fetchall())
    return result > 0


def build_tag_table(conn: Connection):
    cursor = conn.cursor()

    if check_if_table_exists(cursor, "items"):
        return

    cursor.execute("""
        CREATE TABLE tags (
            id integer primary key autoincrement ,
            name text not null unique,
            description text)""")

    conn.commit()


def build_tag_instance_table(conn: Connection):
    cursor = conn.cursor()

    if check_if_table_exists(cursor, "items"):
        return

    cursor.execute("""
        CREATE TABLE tag_instance (
            id integer primary key autoincrement ,
            item integer,
            tag integer,
            foreign key (item) references items(id),
            foreign key (tag) references tags(id) )""")

    conn.commit()


def save_image_path(conn: Connection, path):

    cursor = conn.cursor()
    cursor.execute("""INSERT INTO items(path) values (?);""", (path,))
    row_id = cursor.lastrowid
    conn.commit()

    return Item(path, row_id)


def display_image(image: Image):
    image.show()


def display_item(item: Item):
    image = item.load_image()
    display_image(image)




def main():

    subprocess.call(["rm", "mydatabase.db"])

    conn = sqlite3.connect("mydatabase.db")

    # create a table
    Item.build_table(conn)

    path = "/mnt/nas/datasets/visualgenome/VG_100K/713947.jpg"

    item = save_image_path(conn, path)

    print(item)

    new_item = Item.load_id(conn, 1)

    display_item(new_item)



if __name__ == "__main__":
    main()

