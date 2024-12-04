import os
from ultralytics import YOLO

EPOCH = 2
IMAGE_SIZE = 640

model = YOLO("yolo11m.pt")

# Train the model
train_results = model.train(
    data="dataset/data.yaml",
    epochs=EPOCH,  # number of training epochs
    imgsz=IMAGE_SIZE,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model