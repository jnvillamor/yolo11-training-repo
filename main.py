import os
from ultralytics import YOLO


def train_yolo(epoch, image_size, batch_size, model_path, data_path):
    model = YOLO(model_path)

    # Train the model
    train_results = model.train(
        data=data_path,
        epochs=epoch,
        imgsz=image_size,
        batch=batch_size,
        cache=False,
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model

    return path


if __name__ == "__main__":
    EPOCH = 1
    IMAGE_SIZE = 640
    BATCH_SIZE = 4
    model_path = train_yolo(EPOCH, IMAGE_SIZE, BATCH_SIZE, "yolo11m.pt", "datasets/data.yaml")
    print(f"Model saved at {model_path}")
