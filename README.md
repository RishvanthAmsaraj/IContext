# IContext

**Context-aware computer vision that understands scenes, objects, and human actions.**

IContext combines real-time object detection, pose estimation, and a vision-language model (Florence-2) to not just identify what is in a frame, but to understand the context behind it.

---

## Features

- **Object Detection** -- YOLOv8 with persistent tracking IDs
- **Person Analysis** -- MediaPipe pose, hands, and face mesh
- **Contextual Reasoning** -- Rule-based engine plus Florence-2 VLM descriptions
- **Smart Re-identification** -- Maintains object identity when briefly occluded
- **Hand-Object Interaction** -- Detects how objects are being held or used (typing, holding to ear)
- **Action-Based Reclassification** -- "remote" reclassified as "cell phone" when held to ear
- **Multiple Input Modes** -- Webcam, video files, and static images
- **Snapshot Capture** -- Save annotated frames at any time

---

## Demo

![Demo Output](docs/demo.jpg)

**Example output:**
- **Detected:** `keyboard`, `mouse`, `person`
- **Rule-based context:** "This looks like a computer workstation."
- **Florence-2 description:** "There is a white desk in a room. There is a black laptop on top of the desk. There are two monitors on the desk in front of the laptop."
- **VLM-extracted objects:** desk, laptop, monitors

---

## Installation

### Prerequisites

- **Python 3.11** (tested). Python 3.10 or newer should also work.
- **macOS or Linux** (tested on macOS Darwin 24.6.0)
- A working webcam (required for live webcam mode)

### Setup

```bash
# Clone the repository
git clone https://github.com/RishvanthAmsaraj/IContext.git
cd IContext

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install opencv-python ultralytics numpy python-dotenv mediapipe
pip install transformers accelerate timm einops pillow torch torchvision
```

---

## Usage

### Webcam Mode (Live)

```bash
python src/i_context.py
```

- Press `q` to quit.
- Press `c` to capture and save the current frame.

### Image Mode

```bash
python src/i_context.py --source image --path /path/to/image.jpg
```

Loads a single image, runs the full analysis pipeline, and saves the annotated result to `results/image_analysis.jpg`.

### Video Mode

```bash
python src/i_context.py --source video --path /path/to/video.mp4
```

Processes a video file frame-by-frame using the same pipeline as the webcam mode.

---

## Architecture

```
Input (Webcam / Video / Image)
        |
        v
[YOLOv8 Object Detection + Tracking]
        |
        v
[MediaPipe Pose, Hands, Face Mesh]
        |
        v
[Hand-Object Interaction Analysis]
        |
        v
[Rule-based Context + Florence-2 VLM]
        |
        v
[Annotated Output + Context Description]
```

### Key Components

| Module | Purpose |
|--------|---------|
| `src/i_context.py` | Main pipeline: detection, tracking, interaction, context |
| `src/context_analyzer.py` | Rule-based context inference engine |
| `src/scene_describer.py` | Florence-2 vision-language model wrapper |
| `src/caption_parser.py` | Extracts object nouns from VLM captions |
| `src/detect_objects.py` | Simplified object detection (legacy entry point) |

---

## The "Phone vs Remote" Demo

This is the most illustrative demonstration of the reclassification system:

1. Place a phone flat on a table. It will be detected as `remote` (green box).
2. Pick it up and hold it to your ear.
3. After roughly 2 seconds, the label changes to `cell phone` (yellow box).

The system uses:

- **Hand-object proximity** (distance threshold of 150 pixels)
- **Pose landmarks** (nose position used for "to ear" detection)
- **Finger curl analysis** (typing gesture detection)
- **Voting with decay** (15+ consistent frames required to reclassify)

---

## Context Rules

The rule-based context engine is defined in `src/context_analyzer.py`:

```python
frozenset({"keyboard", "mouse"}): [
    "This looks like a computer workstation.",
    "Someone is likely working at a desk with a keyboard and mouse.",
],
frozenset({"book", "pen", "laptop"}): [
    "Someone was likely doing homework or studying.",
],
# ... additional rules
```

Florence-2 provides richer, natural-language descriptions on top of these rules.

---

## Hybrid Detection: YOLO + Florence-2

YOLOv8 is trained on the COCO dataset (80 object categories) and provides precise bounding boxes. Florence-2 is trained on FLD-5B (5.4 billion annotations) and can describe any visible element.

The system combines both:

- YOLO provides localized detections with bounding boxes.
- Florence-2 provides a natural-language scene description.
- The caption parser extracts object nouns from the Florence-2 output.
- Objects found only by Florence-2 are tagged with `[VLM]` in the display.

**Example result on a desk scene:**
- YOLO: `keyboard`, `mouse`, `person`
- VLM+ : `desk`, `laptop`, `monitors`

---

## Models Used

| Model | Size | Purpose |
|-------|------|---------|
| YOLOv8n | 6 MB | Object detection and tracking |
| MediaPipe | ~10 MB | Pose, hand, and face landmarks |
| Florence-2-base | ~1 GB | Vision-language scene description |

All models run locally. No API calls or cloud dependencies are required.

---

## Performance

On Apple Silicon (M1 / M2 MacBook Pro):

- Object detection: approximately 30 FPS
- MediaPipe processing: approximately 25 FPS
- Florence-2 (per image): approximately 3 to 5 seconds on CPU

Florence-2 is throttled to every 60 frames (~2 seconds at 30 FPS) in live mode to maintain responsiveness.

---

## Tech Stack

- **Python 3.11**
- **OpenCV** -- video capture and image processing
- **Ultralytics YOLOv8** -- object detection and tracking
- **MediaPipe** -- pose, hand, and face landmarks
- **Hugging Face Transformers** -- Florence-2 vision-language model
- **PyTorch** -- deep learning backend

---

## Project Structure

```
IContext/
|-- .gitignore
|-- README.md
|-- src/
|   |-- i_context.py         # Main pipeline
|   |-- context_analyzer.py  # Rule-based context inference
|   |-- scene_describer.py   # Florence-2 VLM wrapper
|   |-- caption_parser.py    # Extract objects from VLM captions
|   |-- detect_objects.py    # Simplified detection (legacy)
|-- docs/
    |-- demo.jpg             # Example output (placeholder)
```

---

## License

MIT License. See `LICENSE` file for details.

---

## Author

**Rishvanth Amsaraj**
- GitHub: [@RishvanthAmsaraj](https://github.com/RishvanthAmsaraj)
- Built as part of an AI/ML research portfolio.