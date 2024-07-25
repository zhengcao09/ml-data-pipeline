import glob
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Parameters
img_formats = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng"]  # acceptable image suffixes
vid_formats = ["mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv"]  # acceptable video suffixes


def split_files(out_path, in_path):  # split training data
    """Splits file names into separate train, test, and val datasets and writes them to prefixed paths."""
    file_names = list(filter(lambda x: len(x) > 0, os.listdir(in_path / "images")))
    file_names = np.array(file_names)
    train, rest = train_test_split(file_names, train_size=0.8, test_size=0.2)
    val, test = train_test_split(rest, test_size=0.5)

    for file in train:
        os.rename(in_path / "images" / file, out_path / "train" / "images" / file)
        txt_file = file.replace("png", "txt")
        os.rename(in_path / "labels" / txt_file, out_path / "train" / "labels" / txt_file)

    for file in val:
        os.rename(in_path / "images" / file, out_path / "valid" / "images" / file)
        txt_file = file.replace("png", "txt")
        os.rename(in_path / "labels" / txt_file, out_path / "valid" / "labels" / txt_file)

    for file in test:
        os.rename(in_path / "images" / file, out_path / "test" / "images" / file)
        txt_file = file.replace("png", "txt")
        os.rename(in_path / "labels" / txt_file, out_path / "test" / "labels" / txt_file)


def make_dirs(dir="datasets/"):
    """Creates a directory with subdirectories 'labels' and 'images', removing existing ones."""
    dir = Path(dir)
    # if dir.exists():
    #     shutil.rmtree(dir)  # delete dir
    for p in (dir / "train" / "images", dir / "train" / "labels", dir / "valid" / "images", dir / "valid" / "labels",
              dir / "test" / "images", dir / "test" / "labels"):
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir


def write_data_data(fname="data.data", nc=80):
    """Writes a Darknet-style .data file with dataset and training configuration."""
    lines = [
        "classes = %g\n" % nc,
        "train =../out/data_train.txt\n",
        "valid =../out/data_test.txt\n",
        "names =../out/data.names\n",
        "backup = backup/\n",
        "eval = coco\n",
    ]

    with open(fname, "a") as f:
        f.writelines(lines)


def image_folder2file(folder="images/"):  # from utils import *; image_folder2file()
    """Generates a txt file listing all images in a specified folder; usage: `image_folder2file('path/to/folder/')`."""
    s = glob.glob(f"{folder}*.*")
    with open(f"{folder[:-1]}.txt", "w") as file:
        for l in s:
            file.write(l + "\n")  # write image list


def create_single_class_dataset(path="../data/sm3"):  # from utils import *; create_single_class_dataset('../data/sm3/')
    """Creates a single-class version of an existing dataset in the specified path."""
    os.system(f"mkdir {path}_1cls")


def flatten_recursive_folders(path="../../Downloads/data/sm4/"):  # from utils import *; flatten_recursive_folders()
    """Flattens nested folders in 'path/images' and 'path/json' into single 'images_flat' and 'json_flat'
    directories.
    """
    idir, _jdir = f"{path}images/", f"{path}json/"
    nidir, njdir = Path(f"{path}images_flat/"), Path(f"{path}json_flat/")
    n = 0

    # Create output folders
    for p in [nidir, njdir]:
        if os.path.exists(p):
            shutil.rmtree(p)  # delete output folder
        os.makedirs(p)  # make new output folder

    for parent, dirs, files in os.walk(idir):
        for f in tqdm(files, desc=parent):
            f = Path(f)
            stem, suffix = f.stem, f.suffix
            if suffix.lower()[1:] in img_formats:
                n += 1
                stem_new = "%g_" % n + stem
                image_new = nidir / (stem_new + suffix)  # converts all formats to *.jpg
                json_new = njdir / f"{stem_new}.json"

                image = parent / f
                json = Path(parent.replace("images", "json")) / str(f).replace(suffix, ".json")

                os.system(f"cp '{json}' '{json_new}'")
                os.system(f"cp '{image}' '{image_new}'")
                # cv2.imwrite(str(image_new), cv2.imread(str(image)))

    print("Flattening complete: %g jsons and images" % n)


def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    """Converts COCO 91-class index (paper) to 80-class index (2014 challenge)."""
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]