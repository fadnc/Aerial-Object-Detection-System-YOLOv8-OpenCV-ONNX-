FROM python:3.10-slim

# opencv needs these system libs to import cleanly in a slim image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Note: cv2.imshow requires a display. In a headless container, pass
# --save output.mp4 to write results to a file instead of showing a window.
CMD ["python", "src/detect.py", "--source", "input.mp4", "--save", "/app/output/annotated.mp4"]
