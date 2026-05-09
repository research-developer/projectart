# Training

YOLO training pipeline for ProjectArt's vision needs.

## Goals, in order

1. **Detect people and cats** — both are in COCO out of the box. Use upstream `yolov8n.pt` (or any size) directly. **No retraining needed for this step.**
2. **Detect hands** — not in COCO. Either fine-tune YOLO on a hand dataset (e.g. EgoHands / OpenImages-Hands) **or** use MediaPipe Hands directly inside the YOLO `person` ROI.
3. **Detect dot-glove markers** — definitely not in COCO. Custom dataset, fine-tune on top of the cat+person checkpoint.

We'll pivot between approaches based on what the live setup needs. Each stage's dataset, script, and result lives in its own subfolder.

## Layout

```
training/
├── README.md                         (this file)
├── datasets/
│   ├── coco_people_cats/             dataset.yaml + symlinks (small COCO subset)
│   ├── hands/                        (planned)
│   └── dot_gloves/                   (planned)
└── scripts/
    ├── verify_yolov8.py              load yolov8n.pt + run inference on a frame
    ├── train_cats_people_subset.sh   fine-tune (small) YOLO on COCO subset
    └── train_dot_gloves.sh           (planned) custom-class fine-tune
```

## Stage 1 — verify person + cat detection (no training)

```bash
# in the project root
pip install -e ".[yolo]"
python training/scripts/verify_yolov8.py /path/to/test.jpg
```

That should print detections including `person` and (if the photo has them) `cat`.
If you have one of the cameras' snapshot URLs in front of you:

```bash
# Pull a snapshot from cam-a and run it
curl -o /tmp/cam-a-snap.jpg http://10.0.0.33/cgi-bin/snapshot.cgi
python training/scripts/verify_yolov8.py /tmp/cam-a-snap.jpg
```

## Stage 2 — fine-tune for tighter cat detection (optional)

If COCO's cat detector misses your specific cats (lighting, angle, breed), fine-tune on a small custom set:

```bash
mkdir -p training/datasets/cats_local/{images,labels}
# Drop ~50 photos in images/, label them with labelImg or similar
# Annotation format: YOLO txt — one line per box: class cx cy w h (normalised)
# Single class: "0 cat" → use class_id 0

bash training/scripts/train_cats_people_subset.sh
```

Output: `training/runs/cats_people_local/weights/best.pt`. Use it via:

```bash
python -m projectart --input gloves \
    --webcam-a rtsp://10.0.0.33/ch0_1.h264 \
    --yolo-weights training/runs/cats_people_local/weights/best.pt
```

## Stage 3 — dot-glove markers (planned)

When the dot gloves are made (see PRD §14):

1. Capture ~200 frames with the gloves in frame, varied lighting.
2. Annotate finger-tip dots: 5 classes per hand (`thumb_dot`, `index_dot`, `middle_dot`, `ring_dot`, `pinky_dot`). Bbox is the dot's bounding square — small (~14 mm).
3. Optionally a 6th `wrist_dot` class for the white back-of-hand HUD anchor.
4. Fine-tune from the cats+people checkpoint (warm-start beats cold).
5. Validate latency on this Mac: target ≥ 30 Hz at the relevant input size.

Reference: the CatMorph YOLO worktree at `/Volumes/research-developer/CatMorph/.worktrees/yolo/` contains the all-integer geometric loss pipeline (training-time speedup). It's an interesting integration to revisit if/when dot-glove training becomes the bottleneck — see `geometric_yolo.py` there. **Not required** for getting Stage 1–3 working.

## Notes

- Keep training **fast and small** — these are Mac-local fine-tunes, not from-scratch runs. `imgsz=384`, `batch=8`, `epochs=10–30` is usually plenty.
- Pin `ultralytics` once locally — version drift causes silent loss-function changes.
- Dataset YAMLs follow the standard Ultralytics format. See `datasets/coco_people_cats/dataset.yaml` once written.
