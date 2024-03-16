from ultralytics import YOLO
from PIL import Image

if __name__ == '__main__':
    # Load a pretrained model (recommended for training)
    model = YOLO("yolov8n.pt")

    # Train the model
    # モデル数100にしたらできた。
    # 学習を増やすために90度回転させた画像も自動で学習させる。
    results = model.train(data='dataset.yaml', epochs=100)
    
