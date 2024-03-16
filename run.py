#学習させた後に使うやつ best.py
#推論モデル

from ultralytics import YOLO
from PIL import Image 

# Load the best weights after training
model = YOLO('./runs/detect/train/weights/best.pt')

# Perform object detection on a test image and save the results
# source=0はWebカメラ
results = model("./test.jpg",show=True,conf=0.2,save=True)