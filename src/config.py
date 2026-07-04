"""
Central configuration for the Aerial Object Detection System.
All paths are resolved relative to the project root so scripts work
no matter what directory they're launched from.
"""
from pathlib import Path
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_YAML = DATA_DIR / "dataset.yaml"

RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

ONNX_MODEL_PATH = MODELS_DIR / "best.onnx"

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
MODEL_NAME = "yolov8n.pt"   # base checkpoint to fine-tune from
IMAGE_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 30
RUN_NAME = "aerial_detector"

# ---------------------------------------------------------------------------
# Inference config
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

# Set to 0 for webcam, or a path/URL for a video file / RTSP stream.
VIDEO_SOURCE = "input.mp4"


def load_class_names(data_yaml: Path = DATA_YAML) -> dict:
    """Load the {id: name} class map from the dataset yaml."""
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    # Normalize keys to int (yaml may load them as int already)
    return {int(k): v for k, v in names.items()}


def find_latest_weights(run_name: str = RUN_NAME) -> Path:
    """
    Find the most recently modified best.pt produced by training.
    Looks under runs/detect/<run_name>*/weights/best.pt and picks the
    newest match. Raises FileNotFoundError with a helpful message if
    nothing has been trained yet.
    """
    candidates = sorted(
        RUNS_DIR.glob(f"{run_name}*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No trained weights found under '{RUNS_DIR}/{run_name}*/weights/best.pt'. "
            f"Run `python src/train.py` first, or pass --weights explicitly."
        )
    return candidates[0]


try:
    CLASS_NAMES = load_class_names()
    NUM_CLASSES = len(CLASS_NAMES)
except FileNotFoundError:
    CLASS_NAMES = {}
    NUM_CLASSES = 0
