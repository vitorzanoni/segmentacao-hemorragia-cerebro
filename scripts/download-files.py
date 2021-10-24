import os
import requests
from multiprocessing import Pool


def download_file(data):
    url, path = data
    path = "CQ500/" + path
    print("Downloading", url, "into", path)
    r = requests.get(url, allow_redirects=True)
    with open(path, 'wb') as f:
        f.write(r.content)
    return True


def main():
    with open("cq500_files.txt", "r") as f:
        urls = [l.rstrip() for l in f.readlines()]

    if not os.path.exists("CQ500"):
        os.mkdir("CQ500")

    names = [l.rsplit("/")[-1] for l in urls]
    urls_and_names = list(zip(urls, names))

    p = Pool(5)
    p.map(download_file, urls_and_names)


if __name__ == "__main__":
    main()
