# Aerial Object Detection System (YOLOv8 + OpenCV + ONNX)

Detects people, vehicles, and other common aerial-scene objects (pedestrian,
people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor —
the full VisDrone taxonomy) in aerial/drone footage using a
YOLOv8 model fine-tuned on your own dataset. Supports two inference paths:

1. **Ultralytics (.pt)** — `src/detect.py`, easiest, needs PyTorch.
2. **ONNX Runtime** — `src/onnx_inference.py`, lighter/faster to deploy, no PyTorch needed at inference time.

## Project structure

```
data/
  dataset.yaml          # class names + paths, edit if you rename folders
  images/{train,val}/   # put your .jpg/.png frames here
  labels/{train,val}/   # matching YOLO-format .txt labels
models/
  best.onnx              # produced by export_onnx.py
output/
  aerial_detector9/      # example results from a prior training run
src/
  config.py             # all paths & hyperparameters live here
  train.py              # fine-tune YOLOv8 on data/dataset.yaml
  export_onnx.py         # convert trained .pt -> .onnx
  detect.py              # live/video inference with Ultralytics
  onnx_inference.py       # live/video inference with ONNX Runtime
  utils.py               # FPS counter, letterbox resize, NMS, drawing
scripts/
  convert_visdrone.py    # VisDrone2019-DET -> YOLO format + class remap
  convert_uavdt.py       # UAVDT-Benchmark-M -> YOLO format
  prepare_roboflow.py    # merge a Roboflow YOLOv8 export, remap by class name
  dataset_stats.py       # per-class instance counts across train/val
gpu_check.py             # quick CUDA availability check

```

## 1. Setup

```bash
pip install -r requirements.txt
python gpu_check.py   # confirms whether training will use GPU or CPU
```

## 2. Build the dataset (VisDrone → UAVDT → Roboflow)

Images go in `data/images/{train,val}/` and matching YOLO-format labels
(one `.txt` per image, same filename) go in `data/labels/{train,val}/`.
Each label line is `<class_id> <x_center> <y_center> <width> <height>`,
all normalized 0-1, with class ids matching `data/dataset.yaml`. This
project uses the full 10-class VisDrone taxonomy for maximum domain range
rather than collapsing to a handful of classes:
```yaml
0: pedestrian
1: people
2: bicycle
3: car
4: van
5: truck
6: tricycle
7: awning-tricycle
8: bus
9: motor
```

The repo ships with empty `images/`/`labels/` folders — you build the
dataset up in three phases using the scripts in `scripts/`. Each converter
copies (doesn't move) files into `data/`, remaps class ids to the table
above, and prefixes filenames so multiple sources can coexist without
collisions.

### Phase 1 — VisDrone2019-DET (primary dataset)

Download from the [official VisDrone repo](https://github.com/VisDrone/VisDrone-Dataset)
(registration required) or a mirror such as [Dataset Ninja](https://datasetninja.com/vis-drone-2019-det).
It natively has all 10 classes above — only "ignored regions" and
"others" are dropped, since those aren't real object classes.

```bash
python scripts/convert_visdrone.py --source /path/to/VisDrone2019-DET-train --split train
python scripts/convert_visdrone.py --source /path/to/VisDrone2019-DET-val --split val
```

### Phase 2 — UAVDT (scale up the vehicle classes)

Download from the [official UAVDT site](https://sites.google.com/view/grli-uavdt)
or a mirror like [Dataset Ninja](https://datasetninja.com/uavdt). It only
covers car/truck/bus (mapped onto ids 3/5/8) — no pedestrian, van,
bicycle, tricycle, or motor, so it purely adds volume to the vehicle
classes you already have from VisDrone. It's video, so `--frame-stride`
skips near-duplicate consecutive frames.

```bash
python scripts/convert_uavdt.py --source /path/to/UAVDT --split train --sequences M0101 M0201 M0301 --frame-stride 5
python scripts/convert_uavdt.py --source /path/to/UAVDT --split val --sequences M0401 --frame-stride 5
```

> UAVDT mirrors vary in folder/file naming — if the script can't find your
> GT files, open one in a text editor and check it matches the column
> order documented at the top of `scripts/convert_uavdt.py`, then adjust
> `GT_SUFFIX` or the paths in the script.

### Phase 3 — Roboflow Universe (fill gaps / add scenes)

Pick a dataset on [Roboflow Universe](https://universe.roboflow.com/) with
classes overlapping the list above, export it as **YOLOv8**, and unzip it.

```bash
python scripts/prepare_roboflow.py --source /path/to/roboflow_export --split train
python scripts/prepare_roboflow.py --source /path/to/roboflow_export --split val
```

The script reads the export's `data.yaml` and remaps by class name; edit
the `ALIASES` dict in `scripts/prepare_roboflow.py` if the Roboflow
dataset uses different wording (e.g. many datasets only have a generic
"person" class, which defaults to "pedestrian" — change that mapping if
"people" fits your source better).

### Check class balance

After merging any combination of the three sources:
```bash
python scripts/dataset_stats.py
```
This prints per-class instance counts for train/val so you can spot
imbalance — the rarer VisDrone-only classes (tricycle, awning-tricycle,
van) will likely need targeted Roboflow searches to bulk up, since UAVDT
doesn't cover them at all.

Manual annotation tools like CVAT or LabelImg also export directly to
this same YOLO format if you want to add your own footage on top.

## 3. Train

```bash
python src/train.py --epochs 30 --imgsz 640 --batch 16
```

Weights are saved to `runs/detect/aerial_detector*/weights/best.pt`.
`src/detect.py` and `src/export_onnx.py` auto-discover the most recent run,
or you can point them at a specific file with `--weights`.

An example run's metrics are kept under `output/aerial_detector9/` for
reference (precision/recall/mAP climbing over 30 epochs on that dataset).

## 4a. Run detection (Ultralytics / .pt)

```bash
python src/detect.py --source input.mp4          # video file
python src/detect.py --source 0                  # webcam
python src/detect.py --source input.mp4 --save output/annotated.mp4
```
Press `Esc` to quit the preview window.

## 4b. Export to ONNX and run detection (ONNX Runtime)

```bash
python src/export_onnx.py                        # writes models/best.onnx
python src/onnx_inference.py --source input.mp4
python src/onnx_inference.py --source input.mp4 --save output/annotated.mp4
```

The ONNX path implements full pre/post-processing itself (letterbox resize,
box decoding, NMS via `cv2.dnn.NMSBoxes`, coordinate rescaling) so it has no
dependency on the `ultralytics` package at inference time — useful for
lighter-weight or edge deployments.

## Docker

```bash
docker build -t aerial-detector .
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output aerial-detector
```

The container has no display, so the default `CMD` writes annotated output
to `/app/output/annotated.mp4` instead of opening a preview window. Mount a
video file into `/app/input.mp4` or edit the `CMD` in the `Dockerfile` to
point at your own source.

## Notes / known limitations

- No dataset ships with this repo — you must supply your own aerial images
  and labels before `train.py` will produce a meaningful model.
- `onnx_inference.py` assumes the standard YOLOv8 ONNX export output shape
  `(1, 4 + num_classes, N)`; if you export with `--dynamic` or a
  non-standard head, adjust `decode_yolov8_output` in `utils.py` accordingly.
- GPU is optional but strongly recommended for training; CPU training on
  a real dataset will be very slow.
