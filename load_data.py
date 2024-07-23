import time

from fathomnet.api import images, boundingboxes
from coco2yolo import convert_coco_json
import os
DIR = "./coco/fathomnet"

concepts = boundingboxes.find_concepts()
species = ['Acanthamunnopsis', 'Acanthascinae sp. 3']
for spe in species:
    cmd = f"fathomnet-generate -c '{spe}' --format coco --img-download '{DIR}/{spe}/images' --output '{DIR}/{spe}'"
    os.system(cmd)
    convert_coco_json(f"{DIR}/{spe}")



# for i in range(1, 3):
#     cmd = f"fathomnet-generate -c '{concepts[i]}' --format coco --img-download '{DIR}/{concepts[i]}/images' --output '{DIR}/{concepts[i]}'"
#     os.system(cmd)
#
#     convert_coco_json(f"{DIR}/{concepts[i]}")

