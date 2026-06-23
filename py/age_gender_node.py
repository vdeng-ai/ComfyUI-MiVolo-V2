import os
import numpy as np
import torch
from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor
from ultralytics import YOLO
import folder_paths
import comfy.model_management

DEFAULT_MIVOLO_MODEL = "iitolstykh/mivolo_v2"
DEFAULT_DETECTOR_MODEL = "iitolstykh/demo_yolov8_detector/yolov8x_person_face.pt"

models_dir = folder_paths.models_dir
mivolo_dir = os.path.join(models_dir, "mivolo")
yolo_dir = os.path.join(models_dir, "yolo")


def ensure_model_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"MiVOLO: failed to create model directory '{path}'. Error: {e}") from e


def local_model_path(base_dir, model_name):
    safe_parts = [part for part in model_name.replace("\\", "/").split("/") if part not in ("", ".")]
    candidate = os.path.abspath(os.path.join(base_dir, *safe_parts))
    base_dir = os.path.abspath(base_dir)
    if os.path.commonpath([base_dir, candidate]) != base_dir:
        raise ValueError(f"MiVOLO: invalid model path '{model_name}'.")
    return candidate


def list_local_mivolo_models():
    if not os.path.isdir(mivolo_dir):
        return []

    model_names = []
    for root, _, files in os.walk(mivolo_dir):
        if "config.json" not in files:
            continue
        rel_path = os.path.relpath(root, mivolo_dir)
        if rel_path != ".":
            model_names.append(rel_path.replace(os.sep, "/"))
    return sorted(model_names)


def crop_box(image, box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(box).astype(int)
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def model_device_and_dtype(model):
    device = getattr(model, "device", None)
    dtype = getattr(model, "dtype", None)
    if device is None or dtype is None:
        param = next(model.parameters(), None)
        if param is not None:
            device = device or param.device
            dtype = dtype or param.dtype
    return device, dtype


def move_tensor(tensor, device, dtype):
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    return tensor.to(**kwargs) if kwargs else tensor


def gender_label(config, gender_idx):
    labels = getattr(config, "gender_id2label", {})
    if isinstance(labels, dict):
        return labels.get(gender_idx, labels.get(str(gender_idx), str(gender_idx)))
    return labels[gender_idx]


def run_mivolo_prediction(model_pack, face_crop, body_crop):
    if face_crop is None and body_crop is None:
        raise ValueError("MiVOLO: at least one face or body crop is required for prediction.")

    model = model_pack["model"]
    image_processor = model_pack["image_processor"]
    config = model_pack["config"]
    device, dtype = model_device_and_dtype(model)

    face_images = [face_crop] if face_crop is not None else [None]
    body_images = [body_crop] if body_crop is not None else [None]

    face_pixel_values = image_processor(images=face_images)["pixel_values"]
    faces_input = move_tensor(face_pixel_values, device, dtype) if face_pixel_values is not None else None

    body_pixel_values = image_processor(images=body_images)["pixel_values"]
    body_input = move_tensor(body_pixel_values, device, dtype) if body_pixel_values is not None else None

    with torch.inference_mode():
        output = model(faces_input=faces_input, body_input=body_input)

    age_float = output.age_output[0].item()
    age_int = int(round(age_float))
    gender_idx = output.gender_class_idx[0].item()
    gender_str = gender_label(config, gender_idx)
    formatted_text = f"a {age_int} year old {gender_str}"

    return (formatted_text, age_int, gender_str)

def tensor_to_np_bgr(image_tensor):
    """
    Converts a ComfyUI RGB IMAGE tensor to a BGR NumPy array.

    ComfyUI IMAGE tensors have shape [B, H, W, C]. Only the first image of a
    batch is processed; if a larger batch is supplied a warning is printed so
    the dropped frames are not silent. Using an explicit index (instead of
    .squeeze()) avoids collapsing the batch dim when B>1 and avoids removing a
    legitimate size-1 spatial/colour dimension.
    """
    arr = image_tensor
    if arr.ndim == 4:
        if arr.shape[0] > 1:
            print(f"MiVOLO: input batch of {arr.shape[0]} images received; only the first is processed.")
        arr = arr[0]
    img_np_rgb = np.clip(255. * arr.cpu().numpy(), 0, 255).astype(np.uint8)
    if img_np_rgb.ndim == 3 and img_np_rgb.shape[2] == 3:
        img_np_bgr = img_np_rgb[:, :, ::-1]
        return img_np_bgr
    return img_np_rgb

class MiVOLOLoader:
    @classmethod
    def INPUT_TYPES(s):
        model_names = list_local_mivolo_models()
        if DEFAULT_MIVOLO_MODEL not in model_names:
            model_names.insert(0, DEFAULT_MIVOLO_MODEL)
        return {"required": {"model_name": (model_names, )}}

    RETURN_TYPES = ("MIVOLO_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "MiVOLO/AgeGender"

    def load_model(self, model_name):
        ensure_model_dir(mivolo_dir)
        local_path = local_model_path(mivolo_dir, model_name)
        model_path = local_path if os.path.isdir(local_path) else model_name
        
        print(f"MiVOLO: Loading age/gender model from: {model_path}")
        device = comfy.model_management.get_torch_device()
        dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForImageClassification.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype).to(device)
            model.eval()
            image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load MiVOLO model '{model_name}'. Error: {e}")
        model_package = {"model": model, "image_processor": image_processor, "config": config}
        return (model_package,)

class MiVOLODetectorLoader:
    @classmethod
    def INPUT_TYPES(s):
        model_names = [DEFAULT_DETECTOR_MODEL]
        if os.path.isdir(yolo_dir):
            local_models = sorted(f for f in os.listdir(yolo_dir) if f.endswith(".pt"))
            if local_models:
                model_names = local_models + [name for name in model_names if name not in local_models]
        return {"required": {"model_name": (model_names, )}}

    RETURN_TYPES = ("DETECTOR_MODEL",)
    FUNCTION = "load_detector"
    CATEGORY = "MiVOLO/AgeGender"

    def load_detector(self, model_name):
        ensure_model_dir(yolo_dir)
        local_path = local_model_path(yolo_dir, model_name)
        local_filename_path = os.path.join(yolo_dir, os.path.basename(model_name))
        
        if os.path.exists(local_path):
            model_path = local_path
        elif os.path.exists(local_filename_path):
            model_path = local_filename_path
        else:
            if "/" not in model_name:
                raise RuntimeError(f"Detector model '{model_name}' was not found in '{yolo_dir}'.")
            try:
                from huggingface_hub import hf_hub_download
                print(f"MiVOLO: Local detector not found. Downloading from Hugging Face Hub: {model_name}")
                repo_id, filename = os.path.split(model_name)
                model_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=yolo_dir)
            except Exception as e:
                raise RuntimeError(f"Failed to download detector '{model_name}'. Error: {e}")
        
        print(f"MiVOLO: Loading detector model from: {model_path}")
        detector = YOLO(model_path)
        return (detector,)

class MiVOLOAgeGenderPredictorWithDetector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mivolo_model": ("MIVOLO_MODEL",),
                "detector_model": ("DETECTOR_MODEL",),
                "image": ("IMAGE",),
                "mode": (["Use persons and faces", "Use persons only", "Use faces only"],),
                "output_selection": (["All", "Largest Person"],),
                "conf_threshold": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 1.0, "step": 0.01}),
                "iou_threshold": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prediction_text", "age", "gender")
    FUNCTION = "predict_with_detector"
    CATEGORY = "MiVOLO/AgeGender"

    def _predict_single(self, model_pack, face_crop, body_crop):
        return run_mivolo_prediction(model_pack, face_crop, body_crop)

    def predict_with_detector(self, mivolo_model, detector_model, image, mode, output_selection, conf_threshold, iou_threshold):
        img_bgr = tensor_to_np_bgr(image)
        results = detector_model(img_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        person_boxes, face_boxes = [], []
        use_persons, use_faces = "persons" in mode, "faces" in mode

        if results and results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls)
                if cls_id == 0 and use_persons:
                    person_boxes.append(box.xyxy[0].cpu().numpy().astype(int))
                elif cls_id == 1 and use_faces:
                    face_boxes.append(box.xyxy[0].cpu().numpy().astype(int))
        
        if output_selection == "Largest Person":
            if not person_boxes:
                return ("No person detected to select the largest.", "", "")
            
            areas = [(p[2] - p[0]) * (p[3] - p[1]) for p in person_boxes]
            largest_idx = np.argmax(areas)
            largest_person_box = person_boxes[largest_idx]

            p_x1, p_y1, p_x2, p_y2 = largest_person_box
            body_crop = crop_box(img_bgr, largest_person_box)
            if body_crop is None:
                return ("Largest person crop was invalid.", "", "")
            face_crop = None

            if use_faces and mode != "Use persons only":
                for f_box in face_boxes:
                    f_x1, f_y1, f_x2, f_y2 = f_box
                    face_center_x, face_center_y = (f_x1 + f_x2) / 2, (f_y1 + f_y2) / 2
                    if (p_x1 < face_center_x < p_x2) and (p_y1 < face_center_y < p_y2):
                        face_crop = crop_box(img_bgr, f_box)
                        break
            
            prediction_text, age, gender = self._predict_single(mivolo_model, face_crop, body_crop)
            return (prediction_text, str(age), gender)

        # --- Default "All" logic ---
        all_results_text, all_ages, all_genders = [], [], []
        associated_face_indices = set()

        if use_persons:
            for i, p_box in enumerate(person_boxes):
                p_x1, p_y1, p_x2, p_y2 = p_box
                body_crop = crop_box(img_bgr, p_box)
                if body_crop is None:
                    continue
                face_crop = None
                
                if use_faces and mode != "Use persons only":
                    best_face_idx = -1
                    for j, f_box in enumerate(face_boxes):
                        if j in associated_face_indices: continue
                        f_x1, f_y1, f_x2, f_y2 = f_box
                        face_center_x, face_center_y = (f_x1 + f_x2) / 2, (f_y1 + f_y2) / 2
                        if (p_x1 < face_center_x < p_x2) and (p_y1 < face_center_y < p_y2):
                            face_crop = crop_box(img_bgr, f_box)
                            best_face_idx = j
                            break
                    if best_face_idx != -1: associated_face_indices.add(best_face_idx)

                prediction_text, age, gender = self._predict_single(mivolo_model, face_crop, body_crop)
                all_results_text.append(f"Person {i+1}: {prediction_text}")
                all_ages.append(str(age))
                all_genders.append(gender)

        if use_faces and mode != "Use persons only":
            for i, f_box in enumerate(face_boxes):
                if i not in associated_face_indices:
                    face_crop = crop_box(img_bgr, f_box)
                    if face_crop is None:
                        continue
                    prediction_text, age, gender = self._predict_single(mivolo_model, face_crop, None)
                    all_results_text.append(f"Unassociated Face {i+1}: {prediction_text}")
                    all_ages.append(str(age))
                    all_genders.append(gender)

        final_text = "\n".join(all_results_text) if all_results_text else "No objects detected or processed."
        ages_str = ",".join(all_ages)
        genders_str = ",".join(all_genders)
        
        return (final_text, ages_str, genders_str)

class MiVOLOAgeGenderPredictorFromCrops:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "mivolo_model": ("MIVOLO_MODEL",), }, "optional": { "face_image": ("IMAGE",), "body_image": ("IMAGE",), }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prediction_text", "age", "gender")
    FUNCTION = "predict_from_crops"
    CATEGORY = "MiVOLO/AgeGender"

    def _predict_single(self, model_pack, face_crop, body_crop):
        return run_mivolo_prediction(model_pack, face_crop, body_crop)

    def predict_from_crops(self, mivolo_model, face_image=None, body_image=None):
        if face_image is None and body_image is None:
            raise ValueError("MiVOLO Predictor (from Crops): At least one input (face_image or body_image) must be provided.")
        
        face_crop_np = tensor_to_np_bgr(face_image) if face_image is not None else None
        body_crop_np = tensor_to_np_bgr(body_image) if body_image is not None else None

        prediction_text, age, gender = self._predict_single(mivolo_model, face_crop_np, body_crop_np)
        
        return (prediction_text, str(age), gender)

NODE_CLASS_MAPPINGS = {
    "MiVOLOLoader": MiVOLOLoader,
    "MiVOLODetectorLoader": MiVOLODetectorLoader,
    "MiVOLOAgeGenderPredictorWithDetector": MiVOLOAgeGenderPredictorWithDetector,
    "MiVOLOAgeGenderPredictorFromCrops": MiVOLOAgeGenderPredictorFromCrops
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiVOLOLoader": "Load MiVOLO Model",
    "MiVOLODetectorLoader": "Load MiVOLO Detector (YOLO)",
    "MiVOLOAgeGenderPredictorWithDetector": "MiVOLO Predictor (with Detector)",
    "MiVOLOAgeGenderPredictorFromCrops": "MiVOLO Predictor (from Crops)"
}
