#!/usr/bin/env python3
"""
Train YOLOv8 model on Waste Detection dataset
"""
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('weights/yolov8n.pt')

# Train the model on the waste detection dataset
results = model.train(
    data='Waste-detect-v-drone-1/data.yaml',
    epochs=50,  # Using 50 epochs for reasonable training time
    imgsz=640,
    batch=16,
    name='waste_detect',
    patience=10,  # Early stopping patience
    save=True,
    device='cpu'  # Change to '0' if GPU is available
)

print("\n" + "="*60)
print("Training completed!")
print(f"Best model saved at: runs/detect/waste_detect/weights/best.pt")
print("="*60)
