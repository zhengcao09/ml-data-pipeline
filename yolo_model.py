from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

results = model.train(data="/Users/zhengcao/PycharmProjects/ml-data-pipeline/data.yaml", epochs=50, imgsz=640) #640

model = YOLO("/Users/zhengcao/PycharmProjects/ml-data-pipeline/runs/detect/train/weights/best.pt")  # pretrained YOLOv8n model

results = model(["/Users/zhengcao/PycharmProjects/ml-data-pipeline/datasets/fathomnet/Acanthamunnopsis/images/f9e5ec07-ecf7-4ebc-a139-67cbc9644905.png"])  # return a list of Results objects

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk