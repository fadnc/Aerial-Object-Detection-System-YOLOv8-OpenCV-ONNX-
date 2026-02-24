from ultralytics import YOLO
from config import MODEL_NAME, IMAGE_SIZE, BATCH_SIZE, EPOCHS

def train():
    model = YOLO(MODEL_NAME)
    
    model.train(
        data="data/dataset.yaml",
        epoxh=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        name="aerial_detector"
    )
    
if __name__ == "__main__":
    train()