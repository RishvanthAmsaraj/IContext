# IContext

**Context-aware computer vision that understands scenes, objects, and human actions.**

IContext combines real-time object detection, pose estimation, and a vision-language model (Florence-2) to not just *see* what's in a frame, but to **understand the context** behind it.

---

## ✨ Features

- 🎯 **Object Detection** — YOLOv8 with persistent tracking IDs
- 👤 **Person Analysis** — MediaPipe pose, hands, and face mesh
- 🧠 **Contextual Reasoning** — Rule-based + Florence-2 VLM descriptions
- 🔄 **Smart Re-identification** — Maintains object identity when briefly occluded
- 🤚 **Hand-Object Interaction** — Detects how you're holding/using objects (typing, holding to ear)
- 🏷️ **Action-Based Reclassification** — "remote" → "cell phone" when held to ear
- 📷 **Multiple Input Modes** — Webcam, video files, and images
- 💾 **Snapshot Capture** — Save annotated frames

---

## 🖼️ Demo

![Demo Output](docs/demo.jpg)

**Example output:**
- **Detected:** `keyboard`, `mouse`, `person`
- **Rule-based context:** "This looks like a computer workstation"
- **Florence-2 description:** "There is a white desk in a room. There is a black laptop on top of the desk. There are two monitors on the desk in front of the laptop."

---

## 🚀 Installation

### Prerequisites
- **Python 3.11** (tested) — 3.10+ should work
- **macOS / Linux** (tested on macOS Darwin 24.6.0)
- Webcam (for live mode)

### Setup

```bash
# Clone the repo
git clone https://github.com/RishvanthAmsaraj/IContext.git
cd IContext

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install opencv-python ultralytics numpy python-dotenv mediapipe
pip install transformers accelerate timm einops pillow torch torchvision
```

---

## 📖 Usage

### Webcam Mode (Live)
```bash
python src/i_context.py
# Press 'q' to quit, 'c' to capture
```

### Image Mode
```bash
python src/i_context.py --source image --path /path/to/image.jpg
```

### Video Mode
```bash
python src/i_context.py --source video --path /path/to/video.mp4
```

---

## 🏗️ Architecture

```
Input (Webcam/Video/Image)
    ↓
[YOLOv8 Object Detection + Tracking]
    ↓
[MediaPipe Pose, Hands, Face]
    ↓
[Hand-Object Interaction Analysis]
    ↓
[Rule-based Context + Florence-2 VLM]
    ↓
[Annotated Output + Context Description]
```

### Key Components

| Module | Purpose |
|--------|---------|
| `src/i_context.py` | Main script with all detection + tracking + context |
| `src/context_analyzer.py` | Rule-based context inference |
| `src/scene_describer.py` | Florence-2 vision-language model wrapper |
| `src/detect_objects.py` | Basic object detection (simpler version) |

---

## 🧪 The "Phone vs Remote" Demo

This is the most impressive demonstration of the reclassification system:

1. Place your phone flat on a table → detected as `remote` (green box)
2. Pick it up and hold it to your ear
3. After ~2 seconds → label changes to `cell phone` (yellow box)
4. The voting system prevents false positives from single-frame errors

The system uses:
- **Hand-object proximity** (distance < 150px)
- **Pose landmarks** (nose position for "to ear" detection)
- **Finger curl analysis** (typing gesture detection)
- **Voting with decay** (15+ consistent frames to reclassify)

---

## 🎯 Context Rules

The system uses rule-based context mapping (in `context_analyzer.py`):

```python
frozenset({"keyboard", "mouse"}): [
    "This looks like a computer workstation.",
    "Someone is likely working at a desk with a keyboard and mouse.",
],
frozenset({"book", "pen", "laptop"}): [
    "Someone was likely doing homework or studying.",
],
# ... more rules
```

Florence-2 provides richer natural language descriptions on top of these rules.

---

## 🤖 Models Used

| Model | Size | Purpose |
|-------|------|---------|
| **YOLOv8n** | 6 MB | Object detection + tracking |
| **MediaPipe** | ~10 MB | Pose/hands/face landmarks |
| **Florence-2-base** | ~1 GB | Vision-language scene description |

All models run locally — no API calls, no cloud dependencies.

---

## 📊 Performance

On M1/M2 MacBook Pro:
- Object detection: ~30 FPS
- MediaPipe: ~25 FPS
- Florence-2 (per image): ~3-5 seconds on CPU

---

## 🛠️ Tech Stack

- **Python 3.11**
- **OpenCV** — Video capture + image processing
- **Ultralytics YOLOv8** — Object detection + tracking
- **MediaPipe** — Pose/hand/face landmarks
- **Hugging Face Transformers** — Florence-2 VLM
- **PyTorch** — Deep learning backend

---

## 📝 License

MIT License — feel free to use this for your own projects!

---

## 🙋 Author

**Rishvanth Amsaraj**
- GitHub: [@RishvanthAmsaraj](https://github.com/RishvanthAmsaraj)
- Built as part of an AI/ML research portfolio project