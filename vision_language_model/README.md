# README.md

## Vision-Language Model with Qwen (VQA)

This project demonstrates **vision-language understanding** using a Qwen-based Vision-Language Model (VLM).

Users provide:

- An input image
- A natural language question

The model then interprets the image and generates a **textual answer grounded in visual content**.

---

## Features

- Qwen2-VL based vision-language inference
- Natural language question answering (VQA)
- Cross-platform support (Windows / Linux / macOS MPS)
- Automatic device selection:
  - CUDA
  - MPS
  - CPU fallback

---

## Project Structure

```bash
vision_language_model/
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
│ ├─ qwen_wrapper/
│ │ └─ load_model.py
│ └─ utils/
│   ├─ config.py
│   ├─ device.py
│   ├─ image_io.py
│   ├─ text.py
│   └─ visualization.py
├─ scripts/
├─ .gitignore
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

#### Install requirements

```bash
pip install -r requirements.txt
```

#### Note

PyTorch should be installed separately depending on your system environment (CUDA / MPS / CPU).

---

### Download Model

No manual download is required.

The Hugging Face model will be automatically downloaded and cached during the first run.

If needed, you can set HF_TOKEN for faster downloads and higher rate limits.

---

## How to Run

### Run the program

```bash
python -m src.main
```

### Interactive Question Input Example

```bash
Question > What is happening in this image?
```

---

## Inputs

<img width="1459" height="815" alt="스크린샷 2026-04-16 오후 3 42 17" src="https://github.com/user-attachments/assets/e4980bf1-84e7-4730-b23a-a88e5db094b2" />
<img width="1464" height="378" alt="스크린샷 2026-04-16 오후 3 42 38" src="https://github.com/user-attachments/assets/15307367-7c09-4f76-9c54-b291dad1dd41" />


- Image
  - Stored in data/input/
  - Path is specified in configs/default.yaml
- Question
  - Entered interactively through CLI

## Outputs

<img width="1345" height="403" alt="스크린샷 2026-04-16 오후 3 44 04" src="https://github.com/user-attachments/assets/5e6296c5-a2b2-4e09-8c63-2b2bc1c3a939" />


- Saved in data/output/
  - result.txt → generated answer

---

## Code Overview

### `main.py`

Controls the overall execution flow of the program.

- Loads config file
- Initializes inference engine
- Starts interactive loop

---

### `inference.py`

Handles the end-to-end vision-language inference process.

- Loads image
- Loads Qwen VLM model
- Receives user questions
- Calls predictor
- Prints and saves results

---

### `predictor.py`

Performs vision-language inference.

- Builds multimodal input (image + question)
- Encodes input using processor
- Runs model inference
- Decodes generated answer

---

### `qwen_wrapper/load_model.py`

Loads the Qwen Vision-Language Model.

- Loads model from Hugging Face
- Automatically downloads weights if needed
- Sets device (CPU / CUDA / MPS)

---

### `utils/config.py`

Loads configuration file.

- Reads YAML config
- Provides structured config dictionary

---

### `utils/device.py`

Selects runtime device.

- Supports auto detection
- CUDA / MPS / CPU fallback

---

### `utils/image_io.py`

Handles image loading and preprocessing.

- Loads image from path
- Converts to RGB
- Resizes image to reduce memory usage

---

### `utils/text.py`

Handles text processing and logging.

- Cleans user input
- Saves question-answer pairs

---

## Pipline

```mermaid
flowchart TD
    A[Start] --> B[Load config.yaml]
    B --> C[Select device]
    C --> D[Load Qwen VLM]
    D --> E[Load input image]
    E --> F[Resize image]
    F --> G[User inputs question]
    G --> H[Build multimodal prompt]
    H --> I[Processor encoding]
    I --> J[Model inference]
    J --> K[Generate answer]
    K --> L[Print answer]
    K --> M[Save answer to file]
    L --> N[End]
    M --> N
```

---
