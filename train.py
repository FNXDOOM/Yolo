from ultralytics import YOLO
import os
import sys
import torch

def main():
    try:
        print("Starting YOLOv8 training...")
        
        # Verify CUDA
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Check if data.yaml exists
        if not os.path.exists("data.yaml"):
            print("ERROR: data.yaml file not found!")
            return
        print("✓ data.yaml found")
        
        # Check if training data directories exist
        if not os.path.exists("css-data/images/train"):
            print("ERROR: Training images directory not found!")
            return
        if not os.path.exists("css-data/images/val"):
            print("ERROR: Validation images directory not found!")
            return
        print("✓ Training and validation directories found")
        
        # Initialize YOLOv8 model
        print("Loading YOLOv8 model...")
        model = YOLO("yolov8s.pt")  # Try yolov8s.pt for more GPU utilization
        print("✓ Model loaded successfully")
        
        # Train the model
        print("Starting training...")
        results = model.train(
            data="data.yaml",      # Path to your data.yaml
            epochs=80,             # Reduced epochs for testing
            imgsz=640,             # Image size (suitable for RTX 3050)
            batch=12,              # Increased batch size for better GPU utilization
            device=0,              # Use GPU 0
            name="yolov8_custom",  # Experiment name
            workers=4,             # Increased workers for faster data loading
            patience=5,            # Early stopping patience
            pretrained=True,       # Use pretrained weights
            verbose=True,          # Show detailed output
            cache='True',          # Cache to disk to reduce RAM usage
            amp=True               # Enable mixed precision training
        )
        
        print("✓ Training completed successfully!")
        
    except Exception as e:
        print(f"ERROR during training: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()