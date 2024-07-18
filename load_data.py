from fathomnet.api import images, boundingboxes
from coco2yolo import convert_coco_json
import os
DIR = "./coco/fathomnet"


concepts = boundingboxes.find_concepts()
images.find_by_concept(concepts[1])

cmd = f"fathomnet-generate -c '{concepts[1]}' --format coco --img-download '{DIR}/images' --output '{DIR}'"

os.system(cmd)

