[简体中文](README_zh.md)

# ComfyUI MiVolo V2 Node


[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Original Project](https://img.shields.io/badge/Original%20Model-iitolstykh/mivolo_v2-blue)](https://huggingface.co/iitolstykh/mivolo_v2)

Use the advanced **MiVolo V2** model directly in ComfyUI for high-precision age and gender prediction!

This project is a ComfyUI wrapper node for the `iitolstykh/mivolo_v2` model. MiVolo is a Transformer-based, multi-input (face and body) model that provides reliable age and gender estimations.

## 🌟 Core Features

* **Age Estimation:** Receives an image (containing a face or body) and outputs the predicted age as a string.
* **Gender Estimation:** Outputs the predicted gender (e.g., Male/Female) as a string.
* **Multi-Person Support:** Automatically processes face and body crops to improve accuracy (based on the original model's capabilities).

## 📊 Model Performance and Candidate Evaluation

> **Support scope:** This project currently supports **MiVOLO v2 only**. FaceAge ClientScan and MiVOLO-Next are listed as research candidates for possible future backends; the tables below do not mean that either model can currently be selected in this ComfyUI node.

### Published LAGENDA Metrics

The following figures were published for the LAGENDA benchmark. Lower age MAE is better; higher CS@5 and gender accuracy are better. They are useful for initial model selection, but are not an independently controlled comparison because the implementations and preprocessing pipelines differ.

| Model | Input | Parameters | Age MAE ↓ | CS@5 ↑ | Gender Accuracy ↑ | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [MiVOLO v2](https://huggingface.co/iitolstykh/mivolo_v2) | Face + body | 28.8M | 3.650 | 74.48% | 97.99% | [Official 2024 paper](https://arxiv.org/abs/2403.02302) |
| [FaceAge ClientScan](https://huggingface.co/TrungTran/faceage_ClientScan) | Face only | 307M | 3.555 | 75.5% | 97.75% | Author-reported model card; not independently reproduced |

### Availability and Integration Readiness

| Model | Weight / License Status | Project Status | Evidence Level |
| --- | --- | --- | --- |
| MiVOLO v2 | Public Hugging Face weights; model card lists Apache-2.0, with upstream terms also requiring review | **Currently supported** | Official papers and model card |
| FaceAge ClientScan | Gated public download after accepting Hugging Face access terms; model card lists Apache-2.0, while the DINOv3 backbone is also subject to the [DINOv3 License](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md) | Candidate; not supported | Author-reported model card |
| MiVOLO-Next | Weights not public; online demo only | Research candidate; not supported | Official repository claim and demo |

## 🖼️ Nodes and Workflow Examples

Example Workflow:
![MiVOLO-V2 Workflow Example](examples/MiVOLO-V2.png)


## 🚀 How to Install

## Compatibility and Safety Notes

* Requires Python 3.10 or newer, matching current ComfyUI baseline support.
* Dependency ranges are intentionally bounded for the current ComfyUI generation and the official MiVOLO V2 model card (`transformers>=4.51.0,<5`, `accelerate>=1.8.1,<2`, `numpy>=1.25.0,<3`, `ultralytics>=8.3.0,<9`). If ComfyUI ships a newer major dependency version, test this node before upgrading a production environment.
* The MiVOLO model is loaded with `trust_remote_code=True` because the Hugging Face model uses custom model code. Only use model repositories you trust, or use a locally reviewed copy for offline deployments.
* The `mivolo` Git dependency is installed from the upstream repository because there is no pinned PyPI package in this project. For fully reproducible deployments, install from a reviewed commit in your own environment.
* ComfyUI batch IMAGE inputs are accepted, but only the first image in the batch is processed.
* Multi-person results return comma-separated `age` and `gender` strings, plus a human-readable `prediction_text`.

### 1. (Recommended) Use ComfyUI Manager
1.  Open ComfyUI Manager.
2.  Click "Install Custom Nodes".
3.  Search for `ComfyUI-MiVolo-V2` and install it.
4.  Restart ComfyUI.

### 2. (Manual) Git Clone
1.  Open a terminal and navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/deng-wei/ComfyUI-MiVolo-V2.git
    ```
3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Restart ComfyUI.

## 📦 Model Installation

This custom node requires two types of models:
1.  **MiVOLO Age/Gender Model** (for prediction)
2.  **YOLO Detector Model** (for finding faces and bodies, optional)

This project supports both automatic downloading and manual placement of models.

### 1. MiVOLO Age/Gender Model (MiVOLOLoader)

This is the main prediction model.

* **Model Name:** `iitolstykh/mivolo_v2`
* **Storage Path:** `ComfyUI/models/mivolo/`

#### Method A: Automatic Download (Recommended)
1.  The code is configured to handle this automatically.
2.  In ComfyUI, add the **"Load MiVOLO Model"** node.
3.  In the `model_name` field, keep the default **`"iitolstykh/mivolo_v2"`** selected.
4.  The first time you run a workflow, the `transformers` library will automatically download this model from Hugging Face and cache it on your system.

#### Method B: Manual Download
If you want to manage models manually or use them in an offline environment:
1.  Visit the Hugging Face repo: [https://huggingface.co/iitolstykh/mivolo_v2](https://huggingface.co/iitolstykh/mivolo_v2)
2.  Download or `git clone` the entire repository.
3.  Ensure all model files (like `config.json`, `pytorch_model.bin`, etc.) are located in a folder named after the model.
4.  Place this folder inside the `mivolo` directory in your ComfyUI `models` directory.

The final path structure should be:
```
ComfyUI/
└── models/
    └── mivolo/
        └── iitolstykh/mivolo_v2/
            ├── config.json
            ├── configuration_mivolo.py
            ├── modeling_mivolo.py
            ├── pytorch_model.bin
            └── ... (and all other files)
```
Once done, the "Load MiVOLO Model" node will automatically detect it in the dropdown list.

The loader detects local MiVOLO folders by looking for `config.json` under `ComfyUI/models/mivolo/`, so custom local model folders are supported when they contain a complete Hugging Face model snapshot.

### 2. YOLO Detector Model (MiVOLODetectorLoader)

This is a `.pt` file used to detect people and faces in an image.

* **Model Name:** `yolov8x_person_face.pt`
* **Hugging Face Repo:** `iitolstykh/demo_yolov8_detector`
* **Storage Path:** `ComfyUI/models/yolo/`

#### Method A: Automatic Download (Recommended)
1.  In ComfyUI, add the **"Load MiVOLO Detector (YOLO)"** node.
2.  Keep the default `model_name` selected: **`"iitolstykh/demo_yolov8_detector/yolov8x_person_face.pt"`**.
3.  The first time you run a workflow, the script will check the `ComfyUI/models/yolo/` folder.
4.  If the `yolov8x_person_face.pt` file is not found, the script will **automatically download it from Hugging Face** and place it in the correct `yolo` folder.

#### Method B: Manual Download
If you prefer to download it manually:
1.  Visit the Hugging Face repo: [https://huggingface.co/iitolstykh/demo_yolov8_detector/tree/main](https://huggingface.co/iitolstykh/demo_yolov8_detector/tree/main)
2.  Download the single file `yolov8x_person_face.pt`.
3.  Place this file in the `yolo` directory under your ComfyUI `models` directory. (Create the `yolo` folder if it doesn't exist).

The final path structure should be:

```
ComfyUI/
└── models/
    └── yolo/
        └── yolov8x_person_face.pt
```

Once done, the "Load MiVOLO Detector (YOLO)" node will be able to load the model immediately.

If the default detector was downloaded automatically, the loader will reuse `ComfyUI/models/yolo/yolov8x_person_face.pt` on later runs instead of downloading it again.

## 💡 Usage Tips

* For best results, ensure the input image is clear and the face/body is visible.
* Supports using pre-cropped faces as input, as well as automatic detection.
* Can be used to analyze AI-generated portraits or for conditional control based on age/gender.

## 📜 Acknowledgments and License

This project is **Adapted Material** based on `iitolstykh/mivolo_v2`.

* **Original Model:** [`iitolstykh/mivolo_v2` (Hugging Face)](https://huggingface.co/iitolstykh/mivolo_v2)
* **Original Papers:**
    * [MiVOLO: Multi-input Transformer for Age and Gender Estimation (2023)](https://arxiv.org/abs/2307.04616)
    * [Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation (2024)](https://arxiv.org/abs/2403.02302)
* **Model Card License:** The Hugging Face model card lists `apache-2.0`. Review the upstream [MiVOLO repository](https://github.com/WildChlamydia/MiVOLO) as well if you redistribute model files or derived assets.

This ComfyUI node project is distributed under the Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0) license as declared in `pyproject.toml`.

This means you are free to use, modify, and distribute this project, provided you give appropriate attribution and share your adaptations under the same license.

## 🐞 Bug Reports

If you encounter any issues or have feature suggestions, please feel free to open an Issue on the "Issues" page!
