# ComfyUI MiVolo V2 节点


[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Original Project](https://img.shields.io/badge/Original%20Model-iitolstykh/mivolo_v2-blue)](https://huggingface.co/iitolstykh/mivolo_v2)

在 ComfyUI 中直接使用先进的 **MiVolo V2** 模型进行高精度的**年龄和性别预测**！

这个项目是 `iitolstykh/mivolo_v2` 模型的一个 ComfyUI 封装节点。MiVolo 是一个基于 Transformer 的多输入（面部和身体）模型，能够提供可靠的年龄和性别估计。

## 🌟 核心功能

* **年龄预测 (Age Estimation):** 接收图像（包含面部或身体），输出预测的年龄的字符串。
* **性别预测 (Gender Estimation):** 输出预测的性别（例如：Male/Female）的字符串。
* **多人输入支持:** 自动处理面部和身体裁剪图，以提高准确性（基于原始模型能力）。

## 📊 模型性能与候选评估

> **支持范围：** 本项目当前仅支持 **MiVOLO v2**。FaceAge ClientScan 和 MiVOLO-Next 仅作为未来可能新增后端的调研候选；下表不代表这两个模型当前可在本 ComfyUI 节点中选择。

### 已发布的 LAGENDA 指标

下列数据均为已发布的 LAGENDA 基准指标。年龄 MAE 越低越好，CS@5 和性别准确率越高越好。由于各模型的实现与预处理流程不同，这些数字可用于初步选型，但不属于独立控制变量的对比实验。

| 模型 | 输入 | 参数量 | 年龄 MAE ↓ | CS@5 ↑ | 性别准确率 ↑ | 证据来源 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [MiVOLO v2](https://huggingface.co/iitolstykh/mivolo_v2) | 人脸 + 身体 | 28.8M | 3.650 | 74.48% | 97.99% | [2024 官方论文](https://arxiv.org/abs/2403.02302) |
| [FaceAge ClientScan](https://huggingface.co/TrungTran/faceage_ClientScan) | 仅人脸 | 307M | 3.555 | 75.5% | 97.75% | 作者在模型卡中自报，尚无独立复现 |

### 权重可用性与集成状态

| 模型 | 权重/许可状态 | 本项目状态 | 证据等级 |
| --- | --- | --- | --- |
| MiVOLO v2 | Hugging Face 权重公开；模型卡标注 Apache-2.0，同时应核对上游条款 | **当前已支持** | 官方论文与模型卡 |
| FaceAge ClientScan | 接受 Hugging Face 访问条款后可下载的门控公开权重；模型卡标注 Apache-2.0，其 DINOv3 backbone 还受 [DINOv3 License](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md) 约束 | 候选，尚未支持 | 作者自报模型卡 |
| MiVOLO-Next | 权重未公开，仅有在线演示 | 调研候选，尚未支持 | 官方仓库声明与演示 |

## 🖼️ 节点和工作流示例

### 示例工作流

![MiVOLO-V2 Workflow Example](examples/MiVOLO-V2.png)

## 🚀 如何安装

## 兼容性与安全说明

* 需要 Python 3.10 或更新版本，与当前 ComfyUI 的基础要求保持一致。
* 依赖范围按当前 ComfyUI 代际和官方 MiVOLO V2 模型卡做了边界约束（`transformers>=4.51.0,<5`、`accelerate>=1.8.1,<2`、`numpy>=1.25.0,<3`、`ultralytics>=8.3.0,<9`）。如果 ComfyUI 升级到新的主版本依赖，建议先在测试环境验证本节点。
* MiVOLO 模型加载时需要 `trust_remote_code=True`，因为 Hugging Face 模型仓库包含自定义模型代码。请只使用可信模型仓库；离线或生产环境建议使用已审查的本地模型副本。
* `mivolo` 依赖来自上游 Git 仓库，本项目没有固定到可验证的 PyPI 包。需要完全可复现部署时，请在自己的环境中安装已审查的固定 commit。
* ComfyUI 的批量 IMAGE 输入可以连接，但本节点只处理 batch 中的第一张图。
* 多人结果中的 `age` 和 `gender` 是逗号分隔字符串，`prediction_text` 是可读文本。

### 1. (推荐) 使用 ComfyUI Manager
1.  打开 ComfyUI Manager。
2.  点击 "Install Custom Nodes"。
3.  搜索 `ComfyUI-MiVolo-V2` 并安装。
4.  重启 ComfyUI。

### 2. (手动) Git Clone
1.  打开终端，进入 ComfyUI 的 `custom_nodes` 目录:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone 本仓库:
    ```bash
    git clone https://github.com/vdeng-ai/ComfyUI-MiVolo-V2.git
    ```
3.  安装依赖:
    ```bash
    pip install -r requirements.txt
    ```
4.  重启 ComfyUI。

-----

## 📦 模型安装说明

本自定义节点需要两种模型：

1.  **MiVOLO 年龄/性别模型** (用于预测)
2.  **YOLO 检测模型** (用于查找人脸和身体，可选)

本项目支持**自动下载**和**手动放置**模型。

### 1\. MiVOLO 年龄/性别模型 (`MiVOLOLoader`)

这是主要的预测模型。

  * **模型名称:** `iitolstykh/mivolo_v2`
  * **存放路径:** `ComfyUI/models/mivolo/`

#### 方式 A：自动下载 (推荐)

代码已配置为自动处理。

1.  在 ComfyUI 中，添加 **"Load MiVOLO Model"** 节点。
2.  在 `model_name` 字段中，**保持选中默认的 `"iitolstykh/mivolo_v2"`**。
3.  第一次运行工作流时，`transformers` 库会自动从 Hugging Face 下载该模型并将其缓存到您的系统中。

#### 方式 B：手动下载

如果您希望手动管理模型，或者在离线环境中使用：

1.  访问 Hugging Face 仓库: [https://huggingface.co/iitolstykh/mivolo\_v2](https://huggingface.co/iitolstykh/mivolo_v2)
2.  将整个仓库下载或 `git clone` 下来。
3.  确保所有模型文件（如 `config.json`, `pytorch_model.bin` 等）都位于一个以模型名称命名的文件夹中。
4.  将该文件夹放置在 ComfyUI 的 `models` 目录下的 `mivolo` 文件夹中。

**最终路径应如下所示:**

```
ComfyUI/
└── models/
    └── mivolo/
        └── iitolstykh/mivolo_v2/
            ├── config.json
            ├── configuration_mivolo.py
            ├── modeling_mivolo.py
            ├── pytorch_model.bin
            └── ... (其他所有文件)
```

完成后，"Load MiVOLO Model" 节点将自动在下拉列表中检测到它。

加载器会在 `ComfyUI/models/mivolo/` 下查找包含 `config.json` 的完整 Hugging Face 模型目录，因此也支持使用其他本地 MiVOLO 模型快照。

-----

### 2\. YOLO 检测模型 (`MiVOLODetectorLoader`)

这是一个 `.pt` 文件，用于在图像中检测人物和面部。

  * **模型名称:** `yolov8x_person_face.pt`
  * **Hugging Face 仓库:** `iitolstykh/demo_yolov8_detector`
  * **存放路径:** `ComfyUI/models/yolo/`

#### 方式 A：自动下载 (推荐)

如果您想自动下载。

1.  在 ComfyUI 中，添加 **"Load MiVOLO Detector (YOLO)"** 节点。
2.  保持选中默认的 `model_name`：`"iitolstykh/demo_yolov8_detector/yolov8x_person_face.pt"`。
3.  第一次运行工作流时，脚本会检查 `ComfyUI/models/yolo/` 文件夹。
4.  如果 `yolov8x_person_face.pt` 文件不存在，脚本将**自动从 Hugging Face 下载它并放置在正确的 `yolo` 文件夹中**。

#### 方式 B：手动下载

如果您想手动下载：

1.  访问 Hugging Face 仓库: [https://huggingface.co/iitolstykh/demo\_yolov8\_detector/tree/main](https://huggingface.co/iitolstykh/demo_yolov8_detector/tree/main)
2.  下载 `yolov8x_person_face.pt` 这一个文件。
3.  将该文件放置在 ComfyUI 的 `models` 目录下的 `yolo` 文件夹中。 (如果 `yolo` 文件夹不存在，请创建它)。

**最终路径应如下所示:**

```
ComfyUI/
└── models/
    └── yolo/
        └── yolov8x_person_face.pt
```

完成后，"Load MiVOLO Detector (YOLO)" 节点将能立即加载该模型。

如果默认检测模型是自动下载的，后续运行会复用 `ComfyUI/models/yolo/yolov8x_person_face.pt`，不会因为默认远程名称而重复下载。

## 💡 使用技巧

* 为了获得最佳效果，请确保输入的图像清晰且人脸/身体可见。
* 支持使用已裁剪的人脸作为输入，也支持自动检测人脸。
* 可以用于分析 AI 生成的人像，或根据年龄/性别进行条件控制。

## 📜 致谢与许可

**本项目是基于 `iitolstykh/mivolo_v2` 的改编材料 (Adapted Material)。**

* **原始模型:** [iitolstykh/mivolo_v2 (Hugging Face)](https://huggingface.co/iitolstykh/mivolo_v2)
* **原始论文:**
    * [MiVOLO: Multi-input Transformer for Age and Gender Estimation (2023)](https://arxiv.org/abs/2307.04616)
    * [Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation (2024)](https://arxiv.org/abs/2403.02302)
* **模型卡许可:** Hugging Face 模型卡标注为 `apache-2.0`。如果需要重新分发模型文件或衍生资源，也请同时核对上游 [MiVOLO 仓库](https://github.com/WildChlamydia/MiVOLO)。

本 ComfyUI 节点项目按 `pyproject.toml` 中声明的 Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0) 许可发布。

这意味着您可以在遵守署名和相同方式共享的前提下，自由地使用、修改和分发本项目。

## 🐞 问题反馈

如果遇到任何问题或有功能建议，请随时在 "Issues" 页面提出！
