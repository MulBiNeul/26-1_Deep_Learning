📌 README.md
# Point-based Segmentation with SAM

This project demonstrates **point-based image segmentation** using the Segment Anything Model (SAM) with an interactive interface.

Users can click on an image to provide:
- Foreground points (object)
- Background points (non-object)

The model then generates a segmentation mask based on these prompts.

---

## 🚀 Features

- SAM-based point prompt segmentation
- Interactive mouse-based input (no manual coordinate typing)
- Cross-platform support (Windows / Linux / macOS MPS)
- Automatic resizing for large images
- Visualization outputs:
  - Prompt image (points)
  - Mask
  - Overlay
  - Combined panel

---

## 📂 Project Structure

```bash
point-based_segmentation/
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
│ ├─ interactive.py
│ ├─ sam_wrapper/
│ │ └─ load_model.py
│ └─ utils/
│ ├─ device.py
│ ├─ image_io.py
│ ├─ points.py
│ └─ visualization.py
├─ scripts/
│ └─ download_checkpoint.py
├─ requirements.txt
└─ README.md
```


---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

📥 Download Model
```bash
python scripts/download_checkpoint.py
```
- No Hugging Face token required
- Model is stored locally in checkpoints/
- 
▶️ How to Run (Interactive Mode)
```bash
python -m src.interactive --config configs/default.yaml
```

🖱️ Controls
```bash
Action	Key / Mouse
Add foreground point	Left click
Add background point	Right click
Run segmentation	Enter
Reset points	r
Save results	s
Exit	q
```

🧠 How It Works
Load SAM model
Load image
User clicks points
Convert points → prompt format
Run SAM inference
Post-process mask
Visualize results

🔄 Pipeline
```
flowchart TD
    A[Start] --> B[Load config.yaml]
    B --> C[Load SAM model]
    C --> D[Load input image]
    D --> E{Resize?}

    E -- Yes --> F[Resize image]
    E -- No --> G[Keep original]

    F --> H[Display image]
    G --> H

    H --> I[User clicks points]
    I --> J{Enter pressed?}

    J -- No --> H
    J -- Yes --> K[Build prompts]

    K --> L[SAM inference]
    L --> M[Post-process mask]
    M --> N[Generate overlay]

    N --> O{Save?}
    O -- Yes --> P[Save outputs]
    O -- No --> Q[Display only]

    P --> R[Continue or exit]
    Q --> R
```
    
🖼️ Outputs

Saved in data/output/:

*_mask.png → binary mask
*_prompt.png → points visualization
*_overlay.png → segmentation overlay
*_panel.png → combined visualization
