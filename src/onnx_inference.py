import cv2
import numpy as np
import onnxruntime as ort
from config import ONNX_MODEL_PATH, VIDEO_SOURCE

def preprocess(frame, img_size=640):
    img = cv2.resize(frame, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def main():
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame)
        outputs = session.run(None, {input_name: input_tensor})

        print("ONNX Inference Successful")

    cap.release()

if __name__ == "__main__":
    main()