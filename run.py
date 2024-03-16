#学習させた後に使うやつ best.py
#推論モデル

from ultralytics import YOLO
from PIL import Image 

# Load the best weights after training
model = YOLO('./runs/detect/train/weights/best.pt')

# Perform object detection on a test image and save the results
# source=0はWebカメラ
results = model(source=0, show=True,conf=0.4,save=True)