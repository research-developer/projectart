#!/usr/bin/env bash
#
# Fine-tune YOLOv8n on a small local cats+people subset.
#
# Expects:
#   training/datasets/cats_local/images/   *.jpg
#   training/datasets/cats_local/labels/   *.txt   (YOLO format)
#   training/datasets/cats_local/dataset.yaml
#
# Output:
#   training/runs/cats_people_local/weights/best.pt
#
# This is a baseline. We can integrate the CatMorph integer geometric
# loss later for training-time speedup — not needed for correctness.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA="${DATA:-datasets/cats_local/dataset.yaml}"
EPOCHS="${EPOCHS:-30}"
IMGSZ="${IMGSZ:-384}"
BATCH="${BATCH:-8}"
WEIGHTS="${WEIGHTS:-yolov8n.pt}"
NAME="${NAME:-cats_people_local}"

if [ ! -f "$DATA" ]; then
  echo "missing $DATA — see training/README.md Stage 2 for setup" >&2
  exit 1
fi

python -c "
from ultralytics import YOLO
model = YOLO('${WEIGHTS}')
model.train(
    data='${DATA}',
    epochs=${EPOCHS},
    imgsz=${IMGSZ},
    batch=${BATCH},
    project='runs',
    name='${NAME}',
    exist_ok=True,
)
"

echo "best weights at: training/runs/${NAME}/weights/best.pt"
