from ultralytics import YOLO

# Load your trained model (from scratch)
model = YOLO(r"C:\Users\gudiy\Desktop\intern\runs\detect\yolov8_custom11\weights\best.pt") # Path to your trained .pt file

# Export to TensorRT (.engine format)
model.export(
    format="engine",  # TensorRT format
    imgsz=640,        # Must match training image size
    half=True,       # FP16 quantization (set True for supported GPUs)
    int8=False,       # INT8 quantization (set True with calibration data if needed)
    dynamic=False,    # Dynamic input shapes (optional)
    workspace=4,      # Workspace size in GiB for TensorRT builder
    data="data.yaml"  # Required for INT8 calibration
)