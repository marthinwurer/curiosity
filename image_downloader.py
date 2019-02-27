import glob
import os
import subprocess
import sys

"""
....urls_file="$raw_data_dir/$cname/urls_$cname.txt"
....images_dir="$raw_data_dir/$cname/IMAGES"
....mkdir -p "$images_dir"
....echo "Class: $cname. Total # of urls: $(cat $urls_file | wc -l)"
....echo "Downloading images..."
....wget -nc -q --timeout=5 --tries=2 -i "$urls_file" -P "$images_dir"
"""


def download_images(base_dir):
    for root, dirs, files in os.walk(base_dir):
        print(root, dirs, files)
        images_dir = os.path.join(root, "IMAGES")
        for file in glob.iglob(root + "/urls.txt"):
            print("downloading url file: %s" % (file,))
            subprocess.call(["wget", "-nc", "-q", "--timeout=5", "--tries=2", "-i", file, "-P", images_dir])


def main():
    download_images(sys.argv[1])


if __name__ == "__main__":
    main()


