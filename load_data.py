import time

from fathomnet.api import images, boundingboxes
from coco2yolo import convert_coco_json
import os
from utils import *
from sklearn.model_selection import train_test_split


DIR = "download/fathomnet"
TEST_PATH = "./datasets/test/images"
TRAIN_PATH = "./datasets/train/images"
VALID_PATH = "./datasets/valid/images"

concepts = boundingboxes.find_concepts()
species = ['Acanthamunnopsis', 'Acanthascinae sp. 3']
make_dirs()

for idx in range(0, len(species)):
    cmd = f"fathomnet-generate -c '{species[idx]}' --format coco --img-download '{DIR}/{species[idx]}/images' --output '{DIR}/{species[idx]}'"
    os.system(cmd)
    specie_dir = Path(f"{DIR}/{species[idx]}")
    convert_coco_json(specie_dir, idx)
    split_files(Path("datasets/"), specie_dir)


if os.path.exists("data.yaml"):
    os.remove("data.yaml")
with open("data.yaml", "a") as file:
    file.write(f"names: \n")
    for spe in species:
        file.write(f"- {spe} \n")
    file.write(f"nc: {len(species)} \n")
    file.write(f"test: {TEST_PATH} \n")
    file.write(f"train: {TRAIN_PATH} \n")
    file.write(f"val: {VALID_PATH} \n")


