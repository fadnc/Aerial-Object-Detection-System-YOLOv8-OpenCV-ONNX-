import cv2
import time
from ultralytics import YOLO
from config import VIDEO_SOURCE

def main():
    model = YOLO("runs/detect/aerial_detector/weights/best.pt")
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame)
        annotated = results[0].plot()
        
        fps = 1 / (time.time() - start)
        
        cv2.putText(
            annotated,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,        
        )
        
        cv2.imshow("Aerial Object Detection", annotated)
        
        if cv2.waitkEY(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()