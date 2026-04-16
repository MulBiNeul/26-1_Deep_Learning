# README.md

## Open-Vocabulary Detection with OWLv2 / OWL-ViT

This project demonstrates **open-vocabulary object detection** using Hugging Face OWL-family models with a text-query interface.

Users provide:

- An input image
- Text queries (for example: `pen, laptop`)

The model then detects objects that match the given text prompts and returns bounding boxes with confidence scores.

---

## Features

- OWLv2 / OWL-ViT based open-vocabulary detection
- Text-query based inference (no fixed class list)
- Cross-platform support (Windows / Linux / macOS MPS)
- Automatic device selection:
  - CUDA
  - MPS
  - CPU fallback
- Visualization outputs:
  - Bounding boxes
  - Predicted labels
  - Confidence scores

---

## Project Structure

```bash
open-vocabulary_detection/
├─ configs/
│ └─ default.yaml
├─ checkpoints/
├─ data/
│ ├─ input/
│ └─ output/
├─ src/
│ ├─ main.py
│ ├─ inference.py
│ ├─ predictor.py
│ ├─ owl_wrapper/
│ │ └─ load_model.py
│ └─ utils/
│   ├─ config.py
│   ├─ device.py
│   ├─ image_io.py
│   ├─ text.py
│   └─ visualization.py
├─ scripts/
│ └─ download_checkpoint.py
├─ requirements.txt
└─ README.md
```

---

## Installation

### Install PyTorch (GPU / CPU / MPS)

#### Windows / Linux (CUDA 11.8 - GPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CPU Only

```bash
pip install torch torchvision torchaudio
```

#### macOS (Apple Silicon - MPS)

```bash
pip install torch torchvision
```

### Install requirements

```bash
pip install -r requirements.txt
```

#### Note

PyTorch should be installed separately depending on your system environment (CUDA / MPS / CPU).

---

## Download Model

```bash
python scripts/download_checkpoint.py --config configs/default.yaml
```

Hugging Face checkpoint is downloaded and cached locally
If needed, you can set HF_TOKEN for faster downloads and higher rate limits

---

## How to Run

### CLI Text Input

```bash
python src/main.py --config configs/default.yaml --text "pen, laptop"
```

### Interactive Text Input

<img width="1468" height="194" alt="스크린샷 2026-04-16 오후 3 20 48" src="https://github.com/user-attachments/assets/bb6df3a6-9399-484b-933b-8a1d0a4a3e51" />

```bash
python src/main.py --config configs/default.yaml
```

### Example input:

```bash
pen, laptop
```

---

## Inputs

<img width="1332" height="751" alt="스크린샷 2026-04-16 오후 3 22 04" src="https://github.com/user-attachments/assets/706aebc4-c91c-487f-9bcf-378889ab1726" />

- Image
  - Stored in data/input/
  - Path is specified in configs/default.yaml
- Text Queries
  - Entered through CLI or interactive prompt

## Outputs

<img width="1008" height="572" alt="스크린샷 2026-04-16 오후 3 22 36" src="https://github.com/user-attachments/assets/1e3213cf-c1a3-4f1e-b637-4e919dd8bb94" />

- Saved in data/output/
  - result.jpg → detection result with bounding boxes and labels

---

## Code Overview

### `main.py`

Controls the overall execution flow of the program.

- Loads config file
- Receives text queries
- Loads model
- Runs inference
- Saves results

---

### `inference.py`

Performs open-vocabulary detection using OWL models.

- Loads image
- Encodes image and text
- Runs model inference
- Post-processes detection results
- Applies NMS

---

### `predictor.py`

Handles user input for text queries.

- Reads CLI input (`--text`)
- Supports interactive input
- Parses text into query list

---

### `owl_wrapper/load_model.py`

Loads OWLv2 or OWL-ViT model from Hugging Face.

- Selects model type
- Loads processor and model
- Moves model to device

---

### `utils/config.py`

Loads YAML configuration file.

---

### `utils/device.py`

Selects the appropriate device (CUDA / MPS / CPU).

---

### `utils/image_io.py`

Handles image loading, resizing, and saving.

---

### `utils/text.py`

Parses and processes text queries.

---

### `utils/visualization.py`

Draws bounding boxes and labels on the image.

---

### `scripts/download_checkpoint.py`

Downloads and caches model checkpoints in advance.

---

## Pipeline

```mermaid
flowchart TD
    A[Start] --> B[Load config.yaml]
    B --> C[Load OWL model]
    C --> D[Select device]
    D --> E[Load input image]
    E --> F[Get text queries]
    F --> G[Build text labels]
    G --> H[Processor encoding]
    H --> I[Model inference]
    I --> J[Post-process detections]
    J --> K[Apply NMS]
    K --> L[Draw bounding boxes]
    L --> M[Save output image]
    M --> N[End]
```

---
