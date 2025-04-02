from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x.pt")  # load a pretrained model

# Export the model to ONNX format
model.export(format="onnx", dynamic=True)  # creates yolov8n.onnx