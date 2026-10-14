"""
CLOSED-LOOP MARINE MONITORING SYSTEM  (v2 - Gemini-verified species ID)

Pipeline
  1. Image quality profiling + U-Net enhancement (with OpenCV safety nets)
  2. Roboflow detection (automatic retry if nothing is found)
  3. Class-aware NMS
  4. Species classification for EVERY fish box (species shown on the box)
  5. AUTOMATIC feedback loop for low-confidence fish (no human input):
       stage 1  single prediction
       stage 2  multi-view re-analysis (10 views: enhanced/original, flips,
                zooms, CLAHE, brightness variants) -> averaged
       stage 3  Grad-CAM-guided re-crop: the model's own attention is used
                to crop tighter on the fish and classify again
  5b. NEW: Gemini vision model looks at every fish crop (taken from the
      ORIGINAL image) and identifies the species. Its answer is fused with
      the local EfficientNet result (agree -> confidence boosted,
      disagree -> the more reliable one wins). A whole-image "school
      summary" is also produced (e.g. "9 x Silver Barb").
  6. Explainable AI: Grad-CAM heat-maps + written justification per fish
  7. Outputs: annotated image, Grad-CAM overlay, Grad-CAM gallery,
     JSON report, HTML report

Run:
    python main.py          (choose an image number, or A for all)
    python main.py --all    (process every image, no menu)
    python main.py --no-gemini   (local model only, for comparison)

Set the Gemini key as an environment variable (do NOT hard-code it):
    Windows CMD        :  set GEMINI_API_KEY=your_key
    Windows PowerShell :  $env:GEMINI_API_KEY="your_key"
    Linux / macOS      :  export GEMINI_API_KEY=your_key
"""

import os
import sys
import json
import html
import argparse
import datetime
import csv
import time
import base64

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "Inputs", "Images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
GROWTH_CSV_PATH = os.path.join(OUTPUT_DIR, "growth_history.csv")
GROWTH_GRAPH_PATH = os.path.join(OUTPUT_DIR, "growth_graph.png")

UNET_MODEL_PATH = os.path.join(BASE_DIR, "uieb_unet.keras")
SPECIES_MODEL_PATH = os.path.join(BASE_DIR, "fish_species_model.keras")

# Optional: if you have class_names.json (list of names in training order)
# next to main.py it is used instead of SPECIES_NAMES below.
CLASS_NAMES_JSON = os.path.join(BASE_DIR, "class_names.json")

# ------------------------------------------------------------
# ROBOFLOW
# Best practice: set an environment variable ROBOFLOW_API_KEY.
# Or paste the key between the quotes below.
# ------------------------------------------------------------
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "PASTE_YOUR_ROBOFLOW_KEY_HERE")

ROBOFLOW_PROJECT = "fish-detection2-jgvqz"
ROBOFLOW_VERSION = 1

ROBOFLOW_CONFIDENCE = 40            # normal detection threshold
ROBOFLOW_FALLBACK_CONFIDENCE = 25   # used by the retry loop if nothing found
ROBOFLOW_OVERLAP = 30

# ------------------------------------------------------------
# GEMINI (vision model used to verify / improve species identification)
# Put the key in the GEMINI_API_KEY environment variable.
# ------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "PASTE_YOUR_GEMINI_KEY_HERE")

# Tried in order; the first one that answers is used.
GEMINI_MODELS = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-3.5-flash"]

GEMINI_BATCH = 6            # fish crops sent per request
GEMINI_CROP_PADDING = 0.15  # a bit more context around the fish for Gemini
GEMINI_MIN_CONF = 0.50      # Gemini answers below this never override the local model
GEMINI_TIMEOUT = 90         # seconds

# The U-Net output is washed-out / blurry and hurts classification.
# False -> detection + species ID use the ORIGINAL image (recommended).
# The enhanced image is still produced and shown in the report.
USE_ENHANCED_FOR_ANALYSIS = False

# ------------------------------------------------------------
# SPECIES DECISION THRESHOLDS
# ------------------------------------------------------------
CONF_HIGH = 0.75        # >= HIGH   : accepted directly (green)
CONF_MEDIUM = 0.45      # >= MEDIUM : shown with "?" (orange)
REFINE_BELOW = CONF_HIGH  # below this the automatic feedback loop starts
FOCUS_THRESHOLD = 0.35  # Grad-CAM level used for the focus re-crop (stage 3)

CROP_PADDING = 0.08     # extra context around each fish box
MIN_CROP_SIZE = 12      # px

# "raw"  -> pass 0-255 pixels (correct for Keras EfficientNet / EfficientNetV2,
#           which contain their own rescaling layer)
# "unit" -> divide by 255 (use only if you trained with rescale=1/255)
SPECIES_INPUT_SCALE = "raw"


# ============================================================
# DETECTOR CLASSES
# ============================================================

# Your Roboflow model outputs the class "fishes" (plural) - that is why the
# species classifier never ran before. All spellings are handled now.
FISH_CLASSES = {
    "fish", "fishes", "small fish", "smallfish"
}

OTHER_MARINE_CLASSES = {
    "crab", "shrimp", "jellyfish", "starfish"
}


# ============================================================
# 31 FISH SPECIES  (must be in the same order as training)
# ============================================================


SPECIES_NAMES = [
    "Bangus",
    "Big Head Carp",
    "Black Spotted Barb",
    "Catfish",
    "Climbing Perch",
    "Fourfinger Threadfin",
    "Freshwater Eel",
    "Glass Perchlet",
    "Goby",
    "Gold Fish",
    "Gourami",
    "Grass Carp",
    "Green Spotted Puffer",
    "Indian Carp",
    "Indo-Pacific Tarpon",
    "Jaguar Gapote",
    "Janitor Fish",
    "Knifefish",
    "Long-Snouted Pipefish",
    "Mosquito Fish",
    "Mudfish",
    "Mullet",
    "Pangasius",
    "Perch",
    "Scat Fish",
    "Silver Barb",
    "Silver Carp",
    "Silver Perch",
    "Snakehead",
    "Tenpounder",
    "Tilapia"
]



def load_species_names():
    if os.path.exists(CLASS_NAMES_JSON):
        try:
            with open(CLASS_NAMES_JSON, "r", encoding="utf-8") as f:
                names = json.load(f)
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=lambda k: int(k))]
            print(f"Loaded {len(names)} class names from class_names.json")
            return list(names)
        except Exception as e:
            print("Could not read class_names.json:", e)
    return list(SPECIES_NAMES)


# ============================================================
# SMALL HELPERS
# ============================================================

def normalize_label(name):
    return str(name).lower().replace("_", " ").replace("-", " ").strip()


def is_fish_class(name):
    n = normalize_label(name)
    # jellyfish / starfish contain the word "fish" - exclude them first
    if any(o in n for o in OTHER_MARINE_CLASSES):
        return False
    return n in FISH_CLASSES or "fish" in n


def to_py(obj):
    """Make numpy types JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def tier_from_conf(conf):
    if conf >= CONF_HIGH:
        return "high"
    if conf >= CONF_MEDIUM:
        return "medium"
    return "low"


TIER_COLORS = {              # BGR
    "high": (0, 200, 0),
    "medium": (0, 165, 255),
    "low": (0, 0, 255),
    "other": (255, 140, 0)
}


# ============================================================
# IMAGE QUALITY PROFILING
# ============================================================

def profile_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    b, g, r = cv2.mean(image)[:3]
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "color_cast": {"blue": round(float(b), 2),
                       "green": round(float(g), 2),
                       "red": round(float(r), 2)}
    }


# ============================================================
# OPENCV ENHANCEMENT (safety net)
# ============================================================

def white_balance(image):
    img = image.astype(np.float32)
    means = [np.mean(img[:, :, c]) for c in range(3)]
    gray = sum(means) / 3.0
    for c in range(3):
        if means[c] > 0:
            img[:, :, c] *= gray / means[c]
    return np.clip(img, 0, 255).astype(np.uint8)


def sharpen(image, amount=0.35, sigma=1.2):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def opencv_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    result = white_balance(result)
    return sharpen(result, amount=0.25)


# ============================================================
# U-NET ENHANCEMENT
# ============================================================

def unet_enhancement(image, model):
    """Returns (enhanced_image, used_unet)"""
    try:
        original_h, original_w = image.shape[:2]

        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        model_h = input_shape[1] or 256
        model_w = input_shape[2] or 256

        resized = cv2.resize(image, (model_w, model_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = np.expand_dims(rgb.astype(np.float32) / 255.0, axis=0)

        prediction = model.predict(x, verbose=0)
        enhanced = prediction[0]

        if enhanced.min() < 0:                    # [-1, 1] output
            enhanced = (enhanced + 1.0) / 2.0
        enhanced = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        enhanced = cv2.resize(enhanced, (original_w, original_h),
                              interpolation=cv2.INTER_CUBIC)
        return enhanced, True

    except Exception as e:
        print("U-Net enhancement failed:", str(e))
        print("Using OpenCV enhancement instead.")
        return opencv_enhancement(image), False


# ============================================================
# ENHANCEMENT CONTROLLER (closed loop on image quality)
# ============================================================

def enhancement_controller(image, unet_model):
    quality = profile_image(image)
    notes = []

    print("\nIMAGE QUALITY")
    print(f"Brightness : {quality['brightness']}")
    print(f"Contrast   : {quality['contrast']}")
    print(f"Sharpness  : {quality['sharpness']}")

    enhanced, used_unet = unet_enhancement(image, unet_model)
    notes.append("U-Net underwater enhancement applied"
                 if used_unet else
                 "U-Net failed -> OpenCV (CLAHE + white balance) used")

    q = profile_image(enhanced)

    # Feedback 1: U-Net works at low resolution, so fine detail can be lost.
    if q["sharpness"] < 0.5 * quality["sharpness"]:
        enhanced = sharpen(enhanced)
        notes.append("Detail recovery (unsharp mask): U-Net output lost "
                     "more than 50% of the original sharpness")

    q = profile_image(enhanced)

    # Feedback 2: still too dark / flat -> bounded OpenCV refinement.
    if q["brightness"] < 25 or q["contrast"] < 12:
        enhanced = opencv_enhancement(enhanced)
        notes.append("OpenCV refinement (CLAHE + white balance): result "
                     "was still too dark / low contrast")

    enhanced_quality = profile_image(enhanced)

    print("\nENHANCED IMAGE QUALITY")
    print(f"Brightness : {enhanced_quality['brightness']}")
    print(f"Contrast   : {enhanced_quality['contrast']}")
    print(f"Sharpness  : {enhanced_quality['sharpness']}")
    for n in notes:
        print(" -", n)

    return enhanced, quality, enhanced_quality, notes


# ============================================================
# ROBOFLOW
# ============================================================

def initialize_roboflow():
    from roboflow import Roboflow   # imported here so the file loads even offline

    print("\nConnecting to Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project(ROBOFLOW_PROJECT)
    model = project.version(ROBOFLOW_VERSION).model
    print("Roboflow connected.")
    return model


def detect_objects(model, image_path, confidence):
    result = model.predict(
        image_path,
        confidence=confidence,
        overlap=ROBOFLOW_OVERLAP
    ).json()

    predictions = result.get("predictions", [])
    img_info = result.get("image", {}) or {}
    try:
        src_w = int(float(img_info.get("width", 0)))
        src_h = int(float(img_info.get("height", 0)))
    except Exception:
        src_w, src_h = 0, 0
    return predictions, src_w, src_h


def normalize_predictions(predictions, image_width, image_height,
                          src_w=0, src_h=0):
    sx = image_width / src_w if src_w else 1.0
    sy = image_height / src_h if src_h else 1.0

    normalized = []
    for pred in predictions:
        class_name = normalize_label(pred.get("class", ""))
        confidence = float(pred.get("confidence", 0))
        x = float(pred.get("x", 0)) * sx
        y = float(pred.get("y", 0)) * sy
        w = float(pred.get("width", 0)) * sx
        h = float(pred.get("height", 0)) * sy

        x1 = int(max(0, min(x - w / 2, image_width - 1)))
        y1 = int(max(0, min(y - h / 2, image_height - 1)))
        x2 = int(max(0, min(x + w / 2, image_width - 1)))
        y2 = int(max(0, min(y + h / 2, image_height - 1)))

        if x2 <= x1 or y2 <= y1:
            continue

        normalized.append({
            "class": class_name, "confidence": confidence,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })
    return normalized


def calculate_iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def class_aware_nms(detections, iou_threshold=0.45):
    final = []
    for class_name in sorted(set(d["class"] for d in detections)):
        items = sorted([d for d in detections if d["class"] == class_name],
                       key=lambda d: d["confidence"], reverse=True)
        while items:
            best = items.pop(0)
            final.append(best)
            items = [c for c in items
                     if calculate_iou(best, c) < iou_threshold]
    final.sort(key=lambda d: d["confidence"], reverse=True)
    return final


def detect_with_feedback(rf_model, enhanced_path, original_path, width, height):
    """Detection feedback loop: if nothing is found, retry with a lower
    threshold and then on the original (un-enhanced) image."""
    attempts = [
        (enhanced_path, ROBOFLOW_CONFIDENCE, "analysis image, normal threshold"),
        (enhanced_path, ROBOFLOW_FALLBACK_CONFIDENCE, "analysis image, lower threshold"),
    ]
    if original_path != enhanced_path:
        attempts.append((original_path, ROBOFLOW_FALLBACK_CONFIDENCE,
                         "original image, lower threshold"))
    log = []
    for path, conf, desc in attempts:
        print(f"\nRunning Roboflow detection ({desc})...")
        preds, sw, sh = detect_objects(rf_model, path, conf)
        dets = normalize_predictions(preds, width, height, sw, sh)
        dets = class_aware_nms(dets, 0.45)
        print(f"Detections after NMS: {len(dets)}")
        log.append({"attempt": desc, "threshold": conf, "found": len(dets)})
        if dets:
            return dets, log
    return [], log


# ============================================================
# SPECIES CLASSIFIER + GRAD-CAM
# ============================================================

def find_cam_layer(model):
    """Last layer (searching backwards) whose output is a 4-D feature map.
    If the backbone is a nested model (e.g. 'efficientnetv2-s'), that nested
    model's output is the feature map - this works as well."""
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer
        except Exception:
            continue
    return None


class SpeciesClassifier:

    def __init__(self, model, class_names):
        self.model = model
        self.class_names = list(class_names)

        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        self.h = int(shape[1] or 224)
        self.w = int(shape[2] or 224)

        try:
            self.num_outputs = int(model.output_shape[-1])
        except Exception:
            self.num_outputs = len(self.class_names)

        if self.num_outputs != len(self.class_names):
            print(f"\nWARNING: model has {self.num_outputs} outputs but "
                  f"{len(self.class_names)} class names are defined.")
            print("Species names may be wrong - check the class order!\n")

        self.cam_layer = find_cam_layer(model)
        self.cam_mode = None          # "direct", "chain" or None
        self.grad_model = None
        self.chain_layers = None
        self.chain_cut = None
        self._setup_gradcam()

        print(f"Species model input size: {self.w}x{self.h}")

    # ---------------- Grad-CAM plumbing ----------------

    def _selfcheck(self, fn):
        """Grad-CAM forward must reproduce the real model output."""
        try:
            x = np.random.RandomState(0).uniform(
                0, 255, (1, self.h, self.w, 3)).astype(np.float32)
            xt = tf.convert_to_tensor(x)
            _, p1 = fn(xt)
            p2 = self.model(xt, training=False)
            if isinstance(p2, (list, tuple)):
                p2 = p2[0]
            return np.allclose(np.asarray(p1), np.asarray(p2), atol=1e-3)
        except Exception:
            return False

    def _setup_gradcam(self):
        if self.cam_layer is None:
            print("No convolutional feature map found - using occlusion maps.")
            return

        # Mode 1: direct sub-model (works for flat models)
        try:
            gm = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[self.cam_layer.output, self.model.output])

            def direct(x, gm=gm):
                conv, preds = gm(x, training=False)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                return conv, preds

            if self._selfcheck(direct):
                self.cam_mode = "direct"
                self._forward = direct
                print(f"Grad-CAM ready (direct) on layer: {self.cam_layer.name}")
                return
        except Exception:
            pass

        # Mode 2: layer chain (works for nested backbones such as
        # EfficientNetV2 inside a wrapper model + GAP + Dense head)
        try:
            layers = [l for l in self.model.layers
                      if not isinstance(l, tf.keras.layers.InputLayer)]
            cut = [i for i, l in enumerate(layers)
                   if l.name == self.cam_layer.name][0]

            def chain(x, layers=layers, cut=cut):
                h = x
                conv = None
                for i, l in enumerate(layers):
                    try:
                        h = l(h, training=False)
                    except TypeError:
                        h = l(h)
                    if i == cut:
                        conv = h
                if isinstance(h, (list, tuple)):
                    h = h[0]
                return conv, h

            if self._selfcheck(chain):
                self.cam_mode = "chain"
                self._forward = chain
                print(f"Grad-CAM ready (chain) on layer: {self.cam_layer.name}")
                return
        except Exception as e:
            print("Grad-CAM chain mode failed:", str(e)[:100])

        print("Grad-CAM could not be attached - using occlusion maps instead.")

    def name_of(self, index):
        if 0 <= index < len(self.class_names):
            return self.class_names[index]
        return f"Class {index}"

    # ---------------- preprocessing / prediction ----------------

    def preprocess(self, crop_bgr):
        resized = cv2.resize(crop_bgr, (self.w, self.h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        if SPECIES_INPUT_SCALE == "unit":
            return rgb / 255.0
        return preprocess_input(rgb)

    def predict_batch(self, crops):
        x = np.stack([self.preprocess(c) for c in crops], axis=0)
        outs = []
        for i in range(0, len(x), 16):
            out = self.model(x[i:i + 16], training=False)
            if isinstance(out, (list, tuple)):
                out = out[0]
            outs.append(np.asarray(out))
        p = np.concatenate(outs, axis=0).astype(np.float64)
        if p.min() < 0 or np.abs(p.sum(axis=1) - 1).max() > 1e-2:   # logits
            e = np.exp(p - p.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
        return p

    # ---------------- explainability ----------------

    def gradcam(self, crop, class_index):
        if self.cam_mode is None:
            return None
        try:
            x = tf.convert_to_tensor(self.preprocess(crop)[None, ...])
            with tf.GradientTape() as tape:
                conv, preds = self._forward(x)
                score = preds[:, class_index]
            grads = tape.gradient(score, conv)
            if grads is None:
                return None
            weights = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam = tf.reduce_sum(conv[0] * weights, axis=-1)
            cam = tf.nn.relu(cam).numpy()
            cam = cam.astype(np.float64)
            if not np.isfinite(cam).all() or cam.max() <= 0:
                return None
            return (cam / cam.max()).astype(np.float32)
        except Exception as e:
            print("Grad-CAM failed:", str(e)[:120])
            return None

    def occlusion_map(self, crop, class_index, grid=6):
        base = cv2.resize(crop, (self.w, self.h))
        variants = [base]
        ch, cw = self.h // grid, self.w // grid
        for gy in range(grid):
            for gx in range(grid):
                v = base.copy()
                v[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw] = 127
                variants.append(v)
        probs = self.predict_batch(variants)[:, class_index]
        drop = np.maximum(0.0, probs[0] - probs[1:]).reshape(grid, grid)
        if drop.max() <= 1e-8:
            return np.ones((grid, grid), dtype=np.float32) * 0.5
        return (drop / drop.max()).astype(np.float32)

    def explain(self, crop, class_index):
        cam = self.gradcam(crop, class_index)
        if cam is not None:
            return cam, "Grad-CAM"
        return self.occlusion_map(crop, class_index), "Occlusion sensitivity"


def describe_attention(cam):
    """Turn a heat-map into plain-English evidence."""
    cam_big = cv2.resize(cam.astype(np.float32), (100, 100))
    mask = cam_big >= 0.5
    coverage = float(mask.mean() * 100)

    ys, xs = np.nonzero(mask)
    cy = float(ys.mean() / 100) if len(ys) else 0.5
    cx = float(xs.mean() / 100) if len(xs) else 0.5
    horiz = "left" if cx < 0.38 else "right" if cx > 0.62 else "central"
    vert = "upper" if cy < 0.38 else "lower" if cy > 0.62 else "middle"
    region = "centre" if (horiz == "central" and vert == "middle") \
        else f"{vert}-{horiz}".replace("middle-", "").replace("-central", "")

    total = float(cam_big.sum()) + 1e-8
    inner = float(cam_big[15:85, 15:85].sum()) / total
    background = round((1 - inner) * 100)

    return {
        "focus_region": region,
        "strong_activation_percent": round(coverage, 1),
        "on_body_percent": round(inner * 100),
        "edge_percent": background,
        "compact": coverage < 35
    }


def build_justification(rec, attention, method, enh_ok=True):
    parts = []
    parts.append(
        f"The detector found an object of class '{rec['detector_class']}' "
        f"with {rec['detector_confidence'] * 100:.0f}% confidence.")

    top = rec["top3"]
    if top:
        s = (f"The local species model's final choice is {top[0]['species']} "
             f"({top[0]['probability'] * 100:.0f}%)")
        if len(top) > 1:
            s += (f", ahead of {top[1]['species']} "
                  f"({top[1]['probability'] * 100:.0f}%); margin "
                  f"{(top[0]['probability'] - top[1]['probability']) * 100:.0f} "
                  f"points")
        parts.append(s + ".")

    if attention:
        s = (f"{method} shows the decision was driven mainly by the "
             f"{attention['focus_region']} part of the fish "
             f"({attention['strong_activation_percent']:.0f}% of the crop "
             f"strongly activated, {'compact' if attention['compact'] else 'broad'} "
             f"pattern).")
        if attention["edge_percent"] > 45:
            s += (" A large share of attention is near the crop border, so "
                  "background or neighbouring fish may have influenced the result.")
        else:
            s += " Attention stays on the fish body, not the background."
        parts.append(s)

    loop = rec.get("feedback_loop", {})
    if loop.get("triggered"):
        steps = "; ".join(f"{s['stage']}: {s['species']} "
                          f"{s['confidence'] * 100:.0f}%"
                          for s in loop["stages"])
        s = (f"Automatic feedback loop (first confidence was below "
             f"{REFINE_BELOW * 100:.0f}%): {steps}. "
             f"{loop['views_agreeing']} views agreed with the final species.")
        if loop.get("changed_prediction"):
            s += " The loop changed the initial prediction."
        elif loop.get("improved"):
            s += " The loop raised the confidence."
        else:
            s += " The loop did not find stronger evidence."
        parts.append(s)

    g = rec.get("gemini")
    if g:
        s = (f"Gemini vision check: {g['common_name']}"
             + (f" ({g['scientific_name']})" if g.get("scientific_name") else "")
             + f" at {g['confidence'] * 100:.0f}% confidence")
        if g.get("key_features"):
            s += f" - evidence: {g['key_features']}"
        parts.append(s.rstrip(".") + ".")
        parts.append(f"Final decision: {rec['species']} "
                     f"(source: {rec['decision_source']}). Grad-CAM above "
                     f"shows where the LOCAL model looked.")

    if rec.get("crop_note"):
        parts.append(rec["crop_note"])

    verdict = {
        "high": "Verdict: HIGH confidence - accepted automatically.",
        "medium": "Verdict: MEDIUM confidence - likely correct (shown with '?').",
        "low": ("Verdict: LOW confidence - best guess shown with '?'. After "
                "automatic re-analysis the evidence is still weak, so this fish "
                "may belong to a species outside the model's training classes.")
    }[rec["tier"]]
    parts.append(verdict)
    return " ".join(parts)



# ============================================================
# GEMINI VISION - species identification / verification
# ============================================================

def gemini_available():
    return bool(GEMINI_API_KEY) and not GEMINI_API_KEY.startswith("PASTE")


def _encode_jpeg(img, max_side=640, min_side=224, quality=92):
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        s = max_side / longest
        img = cv2.resize(img, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA)
    elif longest < min_side:
        s = min_side / longest
        img = cv2.resize(img, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


_GEMINI_STATE = {"dead": False}   # set True after an auth error -> stop calling


def _gemini_post(parts, schema):
    """Send one request. Returns parsed JSON or None. Never raises."""
    if _GEMINI_STATE["dead"]:
        return None
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    headers = {"x-goog-api-key": GEMINI_API_KEY,
               "Content-Type": "application/json"}
    last_err = "unknown error"

    for model in GEMINI_MODELS:
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent")
        for attempt in range(3):
            try:
                r = requests.post(url, headers=headers, json=body,
                                  timeout=GEMINI_TIMEOUT)
            except requests.RequestException as e:
                last_err = f"{model}: network error {str(e)[:80]}"
                time.sleep(2 * (attempt + 1))
                continue

            if r.status_code == 200:
                try:
                    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(text)
                except Exception as e:
                    last_err = f"{model}: unreadable reply ({str(e)[:60]})"
                    break                      # try next model

            last_err = f"{model}: HTTP {r.status_code} {r.text[:160]}"
            if r.status_code in (401, 403):
                print("Gemini rejected the API key:", last_err)
                _GEMINI_STATE["dead"] = True
                return None
            if r.status_code in (429, 500, 503):
                time.sleep(4 * (attempt + 1))  # rate limit / busy -> retry
                continue
            break                              # 400 / 404 -> next model

    print("Gemini request failed:", last_err)
    return None


def _match_class(name, class_names):
    """Map a Gemini species name onto one of the local class names."""
    n = normalize_label(name or "")
    if not n or n == "none":
        return None
    for c in class_names:
        if normalize_label(c) == n:
            return c
    return None


def _clean_gemini_item(item, class_names):
    try:
        conf = float(item.get("confidence", 0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    common = str(item.get("common_name", "")).strip() or "Unknown"
    matched = _match_class(item.get("matched_class"), class_names)
    if matched is None:
        matched = _match_class(common, class_names)
    return {
        "common_name": common,
        "scientific_name": str(item.get("scientific_name", "")).strip(),
        "matched_class": matched,                 # None if outside the 31
        "species": matched or common,             # name used for the label
        "confidence": conf,
        "is_fish": bool(item.get("is_fish", True)),
        "key_features": str(item.get("key_features", "")).strip(),
        "second_choice": str(item.get("second_choice", "")).strip(),
    }


_CROP_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "index": {"type": "INTEGER"},
            "is_fish": {"type": "BOOLEAN"},
            "common_name": {"type": "STRING"},
            "scientific_name": {"type": "STRING"},
            "matched_class": {"type": "STRING"},
            "confidence": {"type": "NUMBER"},
            "key_features": {"type": "STRING"},
            "second_choice": {"type": "STRING"},
        },
        "required": ["index", "is_fish", "common_name", "matched_class",
                     "confidence", "key_features"],
    },
}

_SCENE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "total_fish_visible": {"type": "INTEGER"},
        "groups": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "common_name": {"type": "STRING"},
                    "scientific_name": {"type": "STRING"},
                    "matched_class": {"type": "STRING"},
                    "approx_count": {"type": "INTEGER"},
                    "notes": {"type": "STRING"},
                },
                "required": ["common_name", "matched_class", "approx_count"],
            },
        },
        "overall_comment": {"type": "STRING"},
    },
    "required": ["total_fish_visible", "groups", "overall_comment"],
}


def gemini_classify_crops(items, class_names):
    """items = [(fish_id, crop_bgr), ...]  ->  {fish_id: result_dict}"""
    results = {}
    class_list = ", ".join(class_names)

    for start in range(0, len(items), GEMINI_BATCH):
        chunk = items[start:start + GEMINI_BATCH]
        prompt = (
            "You are an expert ichthyologist identifying fish species in "
            "aquarium / pond / underwater photographs.\n"
            f"You receive {len(chunk)} cropped images labelled 'Image 1' ... "
            f"'Image {len(chunk)}'. In each crop identify the ONE fish that is "
            "centred and fills most of the frame; ignore neighbouring fish.\n"
            "Use visible evidence: body shape, scale pattern, fin shape and "
            "colour (including fin edges), mouth, eye, barbels, markings. "
            "Judge the true body colour, because underwater colour casts and "
            "water haze can mislead.\n"
            f"Reference classes: [{class_list}].\n"
            "If the fish is one of these species (treat common-name synonyms as "
            "a match, e.g. tinfoil barb = Silver Barb), put the reference name "
            "EXACTLY in matched_class. Otherwise set matched_class to 'none' "
            "and give the real common name in common_name.\n"
            "confidence is 0.0-1.0 and must be honest: lower it when the crop "
            "is blurry, tiny, occluded or ambiguous. key_features: one short "
            "sentence naming the features you used. second_choice: the next "
            "most likely species.\n"
            "Return a JSON array with exactly one object per image, in order, "
            "with index = the image number."
        )
        parts = [{"text": prompt}]
        for k, (_fid, crop) in enumerate(chunk, start=1):
            parts.append({"text": f"Image {k}:"})
            parts.append({"inline_data": {"mime_type": "image/jpeg",
                                          "data": _encode_jpeg(crop)}})

        data = _gemini_post(parts, _CROP_SCHEMA)
        if not isinstance(data, list):
            continue
        for item in data:
            try:
                k = int(item.get("index", 0))
            except Exception:
                continue
            if 1 <= k <= len(chunk):
                results[chunk[k - 1][0]] = _clean_gemini_item(item, class_names)
    return results


def gemini_scene_summary(image, class_names):
    """One look at the whole frame: which species are present, how many."""
    class_list = ", ".join(class_names)
    prompt = (
        "You are an expert ichthyologist. Look at this whole photograph and "
        "list the fish species present with approximate counts. Group fish of "
        "the same species together. Judge true body colour (underwater colour "
        f"casts can mislead). Reference classes: [{class_list}]. If a species "
        "is one of them (synonyms count, e.g. tinfoil barb = Silver Barb) put "
        "the reference name EXACTLY in matched_class, otherwise 'none'. "
        "Only count fish you can actually see. overall_comment: one sentence "
        "such as 'a school of one species' or 'two different species groups'."
    )
    parts = [{"text": prompt},
             {"inline_data": {"mime_type": "image/jpeg",
                              "data": _encode_jpeg(image, max_side=1024)}}]
    data = _gemini_post(parts, _SCENE_SCHEMA)
    if not isinstance(data, dict):
        return None
    for g in data.get("groups", []):
        m = _match_class(g.get("matched_class"), class_names) or \
            _match_class(g.get("common_name"), class_names)
        g["species"] = m or g.get("common_name", "Unknown")
    return data


def fuse_decisions(local_species, local_conf, gem):
    """Combine the local EfficientNet result with Gemini's.
    Returns (species, confidence, source)."""
    if not gem or not gem.get("is_fish", True):
        return local_species, local_conf, "local model"

    g_species, g_conf = gem["species"], gem["confidence"]

    if normalize_label(g_species) == normalize_label(local_species):
        boosted = min(0.99, max(g_conf, local_conf) + 0.10)
        return local_species, boosted, "local model + Gemini agree"

    if g_conf >= GEMINI_MIN_CONF:
        return g_species, g_conf, "Gemini (overrode local model)"

    return local_species, local_conf, "local model (Gemini unsure)"


# ============================================================
# CROPS
# ============================================================

def crop_detection(image, det, padding=CROP_PADDING):
    h, w = image.shape[:2]
    bw, bh = det["x2"] - det["x1"], det["y2"] - det["y1"]
    px, py = int(bw * padding), int(bh * padding)
    x1 = max(0, det["x1"] - px)
    y1 = max(0, det["y1"] - py)
    x2 = min(w, det["x2"] + px)
    y2 = min(h, det["y2"] + py)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def center_zoom(crop, keep=0.88):
    h, w = crop.shape[:2]
    dy, dx = int(h * (1 - keep) / 2), int(w * (1 - keep) / 2)
    z = crop[dy:h - dy, dx:w - dx]
    return z if z.size else crop


def build_views(crop_e, crop_o):
    """10 different looks at the same fish for the feedback loop."""
    return [
        crop_e,
        cv2.flip(crop_e, 1),
        crop_o,
        cv2.flip(crop_o, 1),
        center_zoom(crop_e, 0.88),
        center_zoom(crop_e, 0.75),
        center_zoom(crop_o, 0.88),
        opencv_enhancement(crop_o),
        cv2.convertScaleAbs(crop_e, alpha=1.15, beta=10),
        cv2.convertScaleAbs(crop_e, alpha=0.90, beta=-10),
    ]


def focus_crop(crop, cam, margin=0.10):
    """Crop tighter around the area the model attended to (Grad-CAM)."""
    ch, cw = crop.shape[:2]
    cam_r = cv2.resize(cam.astype(np.float32), (cw, ch))
    ys, xs = np.where(cam_r >= FOCUS_THRESHOLD)
    if len(ys) == 0:
        return None
    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
    my, mx = int((y2 - y1) * margin) + 2, int((x2 - x1) * margin) + 2
    y1, y2 = max(0, y1 - my), min(ch, y2 + my)
    x1, x2 = max(0, x1 - mx), min(cw, x2 + mx)
    sub = crop[y1:y2, x1:x2]
    if sub.size == 0 or min(sub.shape[:2]) < MIN_CROP_SIZE:
        return None
    area = sub.shape[0] * sub.shape[1]
    if area < 0.25 * ch * cw or area > 0.95 * ch * cw:
        return None
    return sub


def auto_feedback_summary(records):
    fish = [r for r in records if r["is_fish"]]
    loops = [r for r in fish if r["feedback_loop"].get("triggered")]
    return {
        "fish_total": len(fish),
        "reanalysed_automatically": len(loops),
        "confidence_improved": sum(1 for r in loops if r["feedback_loop"].get("improved")),
        "species_changed": sum(1 for r in loops if r["feedback_loop"].get("changed_prediction")),
        "still_low_confidence": sum(1 for r in fish if r["tier"] == "low"),
    }


# ============================================================
# ANALYSE ONE DETECTION
# ============================================================

def analyse_detection(det, fid, enhanced, original, clf, gem=None):
    """Returns (record, cam, crop_enh, cam_method)"""
    rec = {
        "id": fid,
        "x1": det["x1"], "y1": det["y1"], "x2": det["x2"], "y2": det["y2"],
        "detector_class": det["class"],
        "detector_confidence": det["confidence"],
        "is_fish": is_fish_class(det["class"]),
    }

    # ---------- non-fish objects: keep the detector label ----------
    if not rec["is_fish"]:
        rec.update({
            "label": det["class"].title(),
            "confidence": det["confidence"],
            "tier": "other",
            "species": None,
            "species_confidence": None,
            "top3": [],
            "feedback_loop": {"triggered": False},
            "justification": (
                f"Detected as '{det['class']}' by the Roboflow detector with "
                f"{det['confidence'] * 100:.0f}% confidence. Species "
                f"classification is only applied to fish."),
        })
        return rec, None, None, None

    # ---------- fish: species classification ----------
    crop_e, crop_box = crop_detection(enhanced, det)
    crop_o, _ = crop_detection(original, det)
    rec["crop_box"] = list(crop_box)

    if crop_e.size == 0 or min(crop_e.shape[:2]) < MIN_CROP_SIZE:
        rec.update({
            "label": "Fish (too small)", "confidence": 0.0, "tier": "low",
            "species": None, "species_confidence": 0.0,
            "top3": [], "feedback_loop": {"triggered": False},
            "justification": "The fish region is too small to classify reliably."
        })
        return rec, None, None, None

    # ---- stage 1: single prediction ----
    probs = clf.predict_batch([crop_e])[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    initial_idx, initial_conf = idx, conf

    stages = [{"stage": "initial prediction", "species": clf.name_of(idx),
               "confidence": conf}]
    loop = {"triggered": False, "stages": stages}
    P_views = None

    # ---- automatic feedback loop ----
    if conf < REFINE_BELOW:
        loop["triggered"] = True

        # stage 2: multi-view re-analysis
        views = build_views(crop_e, crop_o)
        P_views = clf.predict_batch(views)
        probs = P_views.mean(axis=0)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        stages.append({"stage": f"multi-view re-analysis ({len(views)} views)",
                       "species": clf.name_of(idx), "confidence": conf})

        # stage 3: Grad-CAM-guided focus re-crop
        if conf < CONF_MEDIUM:
            cam0, _ = clf.explain(crop_e, idx)
            focus = focus_crop(crop_e, cam0)
            if focus is not None:
                F = clf.predict_batch([focus, cv2.flip(focus, 1)])
                probs = 0.5 * probs + 0.5 * F.mean(axis=0)
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                stages.append({"stage": "Grad-CAM focus re-crop",
                               "species": clf.name_of(idx), "confidence": conf})

        agree = int(np.sum(np.argmax(P_views, axis=1) == idx))
        loop.update({
            "views": len(views),
            "views_agreeing": f"{agree}/{len(views)}",
            "improved": conf > initial_conf,
            "changed_prediction": idx != initial_idx,
        })

    order = np.argsort(-probs, kind="stable")[:3]
    top3 = [{"species": clf.name_of(int(i)), "probability": float(probs[i])}
            for i in order]

    # ---- fuse with Gemini's independent identification ----
    local_species, local_conf = clf.name_of(idx), conf
    species, conf, decision_source = fuse_decisions(local_species, local_conf, gem)
    tier = tier_from_conf(conf)

    crop_note = ""
    if min(crop_e.shape[:2]) < 40:
        crop_note = ("Note: the fish is small in the frame, so fine texture "
                     "is limited.")
    elif cv2.Laplacian(cv2.cvtColor(crop_e, cv2.COLOR_BGR2GRAY),
                       cv2.CV_64F).var() < 20:
        crop_note = "Note: this fish region is blurry, which lowers reliability."

    rec.update({
        "species": species,
        "species_confidence": conf,
        "label": species,
        "confidence": conf,
        "tier": tier,
        "top3": top3,
        "feedback_loop": loop,
        "crop_note": crop_note,
        "local_species": local_species,
        "local_confidence": local_conf,
        "gemini": gem,
        "decision_source": decision_source,
    })

    # ---------- Explainable AI (on the final decision) ----------
    cam, method = clf.explain(crop_e, idx)
    attention = describe_attention(cam)
    rec["xai"] = {"method": method, **attention}
    rec["justification"] = build_justification(rec, attention, method)

    return rec, cam, crop_e, method


# ============================================================
# GRAD-CAM VISUALS
# ============================================================

def overlay_cam(crop, cam, alpha=0.45):
    h, w = crop.shape[:2]
    cam_r = cv2.resize(cam.astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)
    cam_r = np.clip(cam_r, 0, 1)
    heat = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(crop, 1 - alpha, heat, alpha, 0)


def make_tile(crop, cam, caption, size=(200, 150)):
    a = cv2.resize(crop, size)
    b = cv2.resize(overlay_cam(crop, cam), size)
    tile = np.hstack([a, b])
    bar = np.full((24, tile.shape[1], 3), 35, dtype=np.uint8)
    cv2.putText(bar, caption[:44], (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, tile])


def make_gallery(tiles, columns=3):
    if not tiles:
        return None
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full((rows * (th + 6), columns * (tw + 6), 3), 20, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, columns)
        sheet[r * (th + 6):r * (th + 6) + th,
              c * (tw + 6):c * (tw + 6) + tw] = t
    return sheet


# ============================================================
# DRAW RESULTS (+ summary panel)
# ============================================================

def create_summary(records):
    counts = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return counts


def draw_results(image, records):
    out = image.copy()
    H, W = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.4, min(0.7, W / 1300))
    thick = 1 if W < 900 else 2

    for r in records:
        color = TIER_COLORS[r["tier"]]
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]

        mark = "" if (r["tier"] in ("high", "other")
                      or r["label"] == "Unidentified fish") else "?"
        text = f"#{r['id']} {r['label']}{mark} {r['confidence'] * 100:.0f}%"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        (tw, th), bl = cv2.getTextSize(text, font, fs, thick)
        lx1 = max(0, min(x1, W - tw - 8))
        if y1 - th - bl - 6 >= 0:
            ly2 = y1
        else:
            ly2 = y1 + th + bl + 6
        ly1 = ly2 - th - bl - 6
        cv2.rectangle(out, (lx1, ly1), (lx1 + tw + 8, ly2), color, -1)
        txt_color = (255, 255, 255) if r["tier"] == "low" else (0, 0, 0)
        cv2.putText(out, text, (lx1 + 4, ly2 - bl - 2), font, fs, txt_color,
                    thick, cv2.LINE_AA)

    # ---------- side panel ----------
    panel_w = 330
    panel = np.full((H, panel_w, 3), 30, dtype=np.uint8)
    y = 28
    cv2.putText(panel, "SUMMARY", (12, y), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    y += 26
    fish_total = sum(1 for r in records if r["is_fish"])
    cv2.putText(panel, f"Objects: {len(records)}   Fish: {fish_total}", (12, y),
                font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    y += 22
    rechecked = sum(1 for r in records if r["feedback_loop"].get("triggered"))
    cv2.putText(panel, f"Auto re-checked: {rechecked}", (12, y),
                font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    y += 24
    counts = create_summary(records)
    max_lines = max(1, (H - y - 110) // 22)
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    for name, c in items[:max_lines]:
        cv2.putText(panel, f"{c} x {name}"[:36], (12, y), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    if len(items) > max_lines:
        cv2.putText(panel, f"+{len(items) - max_lines} more types", (12, y),
                    font, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    ly = H - 90
    for tier, txt in [("high", f"High confidence (>={CONF_HIGH * 100:.0f}%)"),
                      ("medium", f"Medium (>={CONF_MEDIUM * 100:.0f}%) shown with ?"),
                      ("low", "Low - best guess, auto re-checked"),
                      ("other", "Other marine object")]:
        cv2.rectangle(panel, (12, ly - 11), (28, ly + 3), TIER_COLORS[tier], -1)
        cv2.putText(panel, txt, (38, ly), font, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        ly += 20

    return np.hstack([out, panel])


# ============================================================
# GROWTH / APPARENT-SIZE TRACKING
# ============================================================

def update_growth_history(image_name, records):
    """Save per-image mean fish bounding-box size for trend plotting.

    This is apparent pixel area, not calibrated biological length/weight.
    The same fish is not re-identified across images by this detector.
    """
    now = datetime.datetime.now().isoformat(timespec="seconds")
    grouped = {}
    for rec in records:
        if not rec.get("is_fish") or not rec.get("species"):
            continue
        width = max(0, int(rec["x2"]) - int(rec["x1"]))
        height = max(0, int(rec["y2"]) - int(rec["y1"]))
        if width == 0 or height == 0:
            continue
        species = str(rec["species"])
        grouped.setdefault(species, []).append((width, height, width * height))

    fields = ["image", "processed_at", "species", "fish_count",
              "mean_bbox_width_px", "mean_bbox_height_px", "mean_bbox_area_px"]
    rows = []
    for species, measurements in grouped.items():
        count = len(measurements)
        rows.append({
            "image": image_name,
            "processed_at": now,
            "species": species,
            "fish_count": count,
            "mean_bbox_width_px": round(sum(x[0] for x in measurements) / count, 2),
            "mean_bbox_height_px": round(sum(x[1] for x in measurements) / count, 2),
            "mean_bbox_area_px": round(sum(x[2] for x in measurements) / count, 2),
        })

    # Replace prior entries for the same image/species to avoid duplicate points
    # when an image is run more than once.
    existing = []
    if os.path.exists(GROWTH_CSV_PATH):
        try:
            with open(GROWTH_CSV_PATH, "r", newline="", encoding="utf-8") as f:
                existing = list(csv.DictReader(f))
        except Exception as e:
            print("Warning: could not read previous growth history:", e)
    replacements = {(r["image"], r["species"]): r for r in rows}
    existing = [r for r in existing if (r.get("image"), r.get("species")) not in replacements]
    existing.extend(rows)
    existing.sort(key=lambda r: (r.get("processed_at", ""), r.get("image", ""), r.get("species", "")))
    with open(GROWTH_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
    return rows


def create_growth_graph():
    """Plot mean detected bounding-box area by species across image runs."""
    if not os.path.exists(GROWTH_CSV_PATH):
        return False
    try:
        with open(GROWTH_CSV_PATH, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return False
        # Keep one point per image/species; sort by processing timestamp.
        rows.sort(key=lambda r: (r.get("processed_at", ""), r.get("image", "")))
        species_names = sorted({r["species"] for r in rows if r.get("species")})
        if not species_names:
            return False

        fig, ax = plt.subplots(figsize=(12, 6))
        plotted = 0
        for species in species_names:
            sr = [r for r in rows if r.get("species") == species]
            x = [f"{r['image']}\n{r['processed_at'][:10]}" for r in sr]
            y = [float(r.get("mean_bbox_area_px", 0) or 0) for r in sr]
            ax.plot(x, y, marker="o", linewidth=1.8, label=species)
            plotted += 1
        ax.set_title("Fish Apparent-Size Trend Across Processed Images")
        ax.set_xlabel("Image / processing date")
        ax.set_ylabel("Mean bounding-box area (pixels²)")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelrotation=45)
        if plotted <= 12:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
        fig.tight_layout()
        fig.savefig(GROWTH_GRAPH_PATH, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print("Warning: could not create growth graph:", e)
        return False


# ============================================================
# REPORTS
# ============================================================

def write_html_report(path, name, files, original_q, enhanced_q, enh_notes,
                      det_log, records, summary, auto, scene=None):
    e = html.escape
    scene_html = ""
    if scene:
        items = "".join(
            f"<li>~{int(g.get('approx_count', 0))} x {e(str(g.get('species')))}"
            f" <i>({e(str(g.get('scientific_name', '')))})</i></li>"
            for g in scene.get("groups", []))
        scene_html = (f"<h2>2b. Gemini whole-image school summary</h2>"
                      f"<p>Fish visible: {int(scene.get('total_fish_visible', 0))}"
                      f" - {e(str(scene.get('overall_comment', '')))}</p>"
                      f"<ul>{items}</ul>")
    rows = []
    for r in records:
        tile = files["tiles"].get(r["id"])
        img = f'<img src="{e(tile)}" width="400">' if tile else ""
        alts = "<br>".join(
            f"{e(t['species'])}: {t['probability'] * 100:.0f}%" for t in r["top3"])
        rows.append(
            f"<tr><td>{r['id']}</td><td>{img}</td>"
            f"<td><b>{e(r['label'])}</b><br>{r['confidence'] * 100:.0f}%<br>"
            f"<span class='t {r['tier']}'>{r['tier']}</span><br>"
            f"<small>{e(str(r.get('decision_source', '')))}</small></td>"
            f"<td>{alts}</td><td>{e(r['justification'])}</td></tr>")

    summary_li = "".join(f"<li>{c} x {e(n)}</li>" for n, c in
                         sorted(summary.items(), key=lambda kv: -kv[1]))
    notes_li = "".join(f"<li>{e(n)}</li>" for n in enh_notes)
    log_li = "".join(
        f"<li>{e(d['attempt'])}: {d['found']} found</li>" for d in det_log)
    auto_html = (
        f"<ul><li>Fish detected: {auto['fish_total']}</li>"
        f"<li>Automatically re-analysed (low confidence): "
        f"{auto['reanalysed_automatically']}</li>"
        f"<li>Confidence improved by the loop: {auto['confidence_improved']}</li>"
        f"<li>Species changed by the loop: {auto['species_changed']}</li>"
        f"<li>Still low confidence after the loop: "
        f"{auto['still_low_confidence']}</li></ul>")

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Marine Monitoring Report - {e(name)}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#10161c;color:#e8eef3}}
h1,h2{{color:#7fd1ff}} table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #2b3946;padding:8px;vertical-align:top;font-size:14px}}
th{{background:#1b2630}} img{{max-width:100%;border-radius:6px}}
.t{{padding:2px 8px;border-radius:10px;color:#000}}
.high{{background:#3ddc68}} .medium{{background:#ffb23d}} .low{{background:#ff5d5d}}
.other{{background:#5da8ff}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
</style></head><body>
<h1>Closed-Loop Marine Monitoring Report</h1>
<p>Image: <b>{e(name)}</b> &nbsp;|&nbsp; Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}</p>
<h2>1. Result</h2>
<img src="{e(files['result'])}">
<h2>2. Counts</h2><ul>{summary_li}</ul>
{scene_html}
<h2>3. Fish apparent-size trend</h2>
{("<img src='" + e(files["growth_graph"]) + "' alt='Fish apparent-size trend graph'>") if files.get("growth_graph") else "<p>No fish size measurements available yet.</p>"}
<p>Uses mean detector bounding-box area in pixels as an apparent-size proxy. It is not a calibrated measure of biological growth and does not identify the same individual fish across images.</p>
<h2>4. Image enhancement</h2>
<p>Original: brightness {original_q['brightness']}, contrast {original_q['contrast']},
sharpness {original_q['sharpness']}<br>
Enhanced: brightness {enhanced_q['brightness']}, contrast {enhanced_q['contrast']},
sharpness {enhanced_q['sharpness']}</p><ul>{notes_li}</ul>
<div class="grid"><div><h3>Original</h3><img src="{e(files['original'])}"></div>
<div><h3>Enhanced</h3><img src="{e(files['enhanced'])}"></div></div>
<h2>5. Detection feedback loop</h2><ul>{log_li}</ul>
<h2>6. Explainable AI - Grad-CAM</h2>
<p>Red/yellow areas show the image regions that pushed the model towards its decision.</p>
<img src="{e(files['gradcam_full'])}">
<h2>7. Per-fish decisions and justification</h2>
<table><tr><th>#</th><th>Crop | Grad-CAM</th><th>Decision</th><th>Top-3</th>
<th>AI justification</th></tr>{''.join(rows)}</table>
<h2>8. Automatic species feedback loop</h2>{auto_html}
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(image_path, unet_model, clf, roboflow_model,
                  use_gemini=False):

    print("\n" + "=" * 65)
    print(f"PROCESSING: {os.path.basename(image_path)}")
    print("=" * 65)

    image = cv2.imread(image_path)
    if image is None:
        print("ERROR: Could not read image.")
        return

    height, width = image.shape[:2]
    filename = os.path.basename(image_path)
    name, extension = os.path.splitext(filename)

    out_dir = os.path.join(OUTPUT_DIR, name)
    cam_dir = os.path.join(out_dir, "gradcam_crops")
    os.makedirs(cam_dir, exist_ok=True)

    # ---------------- 1. enhancement ----------------
    enhanced, original_q, enhanced_q, enh_notes = enhancement_controller(
        image, unet_model)

    original_path = os.path.join(out_dir, f"{name}_original{extension}")
    enhanced_path = os.path.join(out_dir, f"{name}_enhanced{extension}")
    cv2.imwrite(original_path, image)
    cv2.imwrite(enhanced_path, enhanced)

    # Which picture is used for detection + species ID?
    # The original is usually better than the washed-out U-Net output.
    if USE_ENHANCED_FOR_ANALYSIS:
        work, work_path = enhanced, enhanced_path
    else:
        work, work_path = image, original_path
    print(f"\nAnalysis image: {'ENHANCED' if USE_ENHANCED_FOR_ANALYSIS else 'ORIGINAL'}")

    # ---------------- 2. detection (with retry loop) ----------------
    detections, det_log = detect_with_feedback(
        roboflow_model, work_path, original_path, width, height)

    
    gem_results, scene = {}, None
    if use_gemini and detections:
        
        items = []
        for fid, det in enumerate(detections, start=1):
            if is_fish_class(det["class"]):
                crop, _ = crop_detection(image, det, GEMINI_CROP_PADDING)
                if crop.size and min(crop.shape[:2]) >= MIN_CROP_SIZE:
                    items.append((fid, crop))
        gem_results = gemini_classify_crops(items, clf.class_names)
        scene = gemini_scene_summary(image, clf.class_names)
        print(f"Gemini answered for {len(gem_results)} of {len(items)} fish.")
        if scene:
            print("\nGEMINI SCHOOL SUMMARY")
            print("-" * 40)
            for g in scene.get("groups", []):
                print(f"  ~{g.get('approx_count', '?')} x {g.get('species')}")
            print("  " + str(scene.get("overall_comment", "")))

    # ---------------- 3. species + explainability ----------------
    print("\nClassifying species and generating Grad-CAM...")
    records, cams, tiles_img, tile_files = [], [], [], {}
    for fid, det in enumerate(detections, start=1):
        rec, cam, crop_e, method = analyse_detection(
            det, fid, work, image, clf, gem_results.get(fid))
        records.append(rec)
        if cam is not None:
            cams.append((cam, rec["crop_box"]))
            caption = f"#{fid} {rec['label']} {rec['confidence'] * 100:.0f}%"
            tile = make_tile(crop_e, cam, caption)
            tile_name = f"fish_{fid:02d}.jpg"
            cv2.imwrite(os.path.join(cam_dir, tile_name), tile)
            tile_files[fid] = f"gradcam_crops/{tile_name}"
            tiles_img.append(tile)
        extra = ""
        if rec.get("gemini"):
            extra = (f"  (local: {rec['local_species']} "
                     f"{rec['local_confidence'] * 100:.0f}% | "
                     f"{rec['decision_source']})")
        print(f"  #{fid:02d} {rec['label']:<24} {rec['confidence'] * 100:5.1f}%"
              f"  [{rec['tier']}]{extra}")

    # ---------------- full-frame Grad-CAM ----------------
    full_cam = np.zeros((height, width), dtype=np.float32)
    for cam, (x1, y1, x2, y2) in cams:
        resized = cv2.resize(cam.astype(np.float32), (x2 - x1, y2 - y1),
                             interpolation=cv2.INTER_CUBIC)
        full_cam[y1:y2, x1:x2] = np.maximum(full_cam[y1:y2, x1:x2], resized)
    gradcam_full = overlay_cam(work, full_cam, alpha=0.5)
    gradcam_full_path = os.path.join(out_dir, f"{name}_gradcam{extension}")
    cv2.imwrite(gradcam_full_path, gradcam_full)

    gallery = make_gallery(tiles_img, columns=3)
    if gallery is not None:
        cv2.imwrite(os.path.join(out_dir, f"{name}_gradcam_gallery.jpg"), gallery)

    # ---------------- 4. outputs ----------------
    result_image = draw_results(work, records)
    result_path = os.path.join(out_dir, f"{name}_result{extension}")
    cv2.imwrite(result_path, result_image)

    summary = create_summary(records)
    auto = auto_feedback_summary(records)

    # Persist per-image apparent-size measurements and refresh the trend graph.
    growth_rows = update_growth_history(filename, records)
    growth_graph_created = create_growth_graph()
    print("\nFISH SIZE TREND")
    print(f"Species measurements saved: {len(growth_rows)}")
    print(f"Growth-history CSV: {GROWTH_CSV_PATH}")
    print(f"Apparent-size graph: {GROWTH_GRAPH_PATH if growth_graph_created else 'not created (no fish measurements yet)'}")
    print("Note: bounding-box pixel area is only an apparent-size proxy, not true biological growth.")

    print("\nDETECTION SUMMARY")
    print("-" * 40)
    for class_name, count in sorted(summary.items(), key=lambda kv: -kv[1]):
        print(f"{class_name}: {count}")
    print(f"\nTotal objects: {len(records)}")

    print("\nAUTOMATIC FEEDBACK LOOP")
    print("-" * 40)
    print(f"Fish re-analysed automatically : {auto['reanalysed_automatically']}"
          f" of {auto['fish_total']}")
    print(f"Confidence improved            : {auto['confidence_improved']}")
    print(f"Species changed by the loop    : {auto['species_changed']}")
    print(f"Still low confidence           : {auto['still_low_confidence']}")

    print("\nAI JUSTIFICATION (first 5 objects)")
    print("-" * 40)
    for r in records[:5]:
        print(f"#{r['id']} {r['label']}: {r['justification']}\n")

    report = {
        "image": filename,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "image_quality": {"original": original_q, "enhanced": enhanced_q,
                          "enhancement_steps": enh_notes},
        "detection_feedback_loop": det_log,
        "gemini_used": bool(gem_results),
        "analysis_image": "enhanced" if USE_ENHANCED_FOR_ANALYSIS else "original",
        "gemini_school_summary": scene,
        "total_objects": len(records),
        "counts": summary,
        "automatic_feedback_loop": auto,
        "growth_tracking": {"csv": os.path.basename(GROWTH_CSV_PATH),
                            "graph": os.path.basename(GROWTH_GRAPH_PATH) if growth_graph_created else None,
                            "metric": "mean fish bounding-box area in pixels; apparent size only"},
        "detections": records
    }
    report_path = os.path.join(out_dir, f"{name}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(to_py(report), f, indent=4)

    html_path = os.path.join(out_dir, f"{name}_report.html")
    write_html_report(
        html_path, filename,
        {"result": os.path.basename(result_path),
         "original": os.path.basename(original_path),
         "enhanced": os.path.basename(enhanced_path),
         "gradcam_full": os.path.basename(gradcam_full_path),
         "growth_graph": os.path.relpath(GROWTH_GRAPH_PATH, out_dir).replace(os.sep, "/") if growth_graph_created else "",
         "tiles": tile_files},
        original_q, enhanced_q, enh_notes, det_log, records, summary, auto,
        scene)

    print("OUTPUT FOLDER:", out_dir)
    print("  Final result      :", os.path.basename(result_path))
    print("  Grad-CAM overlay  :", os.path.basename(gradcam_full_path))
    print("  Grad-CAM gallery  :", f"{name}_gradcam_gallery.jpg")
    print("  JSON report       :", os.path.basename(report_path))
    print("  HTML report       :", os.path.basename(html_path))
    print("  Growth graph      :", GROWTH_GRAPH_PATH if growth_graph_created else "not available")
    print("  Growth history    :", GROWTH_CSV_PATH)
    print("  Growth graph      :", GROWTH_GRAPH_PATH if growth_graph_created else "not available")
    print("  Growth history    :", GROWTH_CSV_PATH)

    print("\n" + "=" * 65)
    print("PROCESSING COMPLETE")
    print("=" * 65)


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():
    print("\nLoading U-Net enhancement model...")
    unet_model = tf.keras.models.load_model(UNET_MODEL_PATH, compile=False)
    print("U-Net loaded.")

    print("\nLoading fish species model...")
    species_model = tf.keras.models.load_model(SPECIES_MODEL_PATH, compile=False)
    print("Fish species model loaded.")

    clf = SpeciesClassifier(species_model, load_species_names())
    return unet_model, clf


# ============================================================
# FILES
# ============================================================

def get_image_files():
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"\nCreated input folder:\n{INPUT_DIR}")
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(exts))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="process all images without asking")
    parser.add_argument("--no-gemini", action="store_true",
                        help="use only the local species model")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("CLOSED-LOOP MARINE MONITORING SYSTEM")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for file_path in (UNET_MODEL_PATH, SPECIES_MODEL_PATH):
        if not os.path.exists(file_path):
            print(f"\nERROR: Model not found:\n{file_path}")
            return

    try:
        unet_model, clf = load_models()
    except Exception as e:
        print("\nERROR loading models:")
        print(e)
        return

    if ROBOFLOW_API_KEY.startswith("PASTE"):
        print("\nERROR: put your Roboflow API key in ROBOFLOW_API_KEY "
              "(or set the ROBOFLOW_API_KEY environment variable).")
        return

    try:
        roboflow_model = initialize_roboflow()
    except Exception as e:
        print("\nERROR connecting to Roboflow:")
        print(e)
        print("\nCheck your Roboflow API key, project name and internet connection.")
        return

    use_gemini = gemini_available() and not args.no_gemini
    if args.no_gemini:
        print("\nGemini disabled (--no-gemini): local species model only.")
    elif not use_gemini:
        print("\nWARNING: GEMINI_API_KEY not set -> running with the local "
              "species model only (less accurate).")
    else:
        print(f"\nGemini enabled (models tried in order: "
              f"{', '.join(GEMINI_MODELS)})")

    image_files = get_image_files()
    if not image_files:
        print("\nNo images found in:")
        print(INPUT_DIR)
        return

    if args.all:
        selected = image_files
    else:
        print("\nAVAILABLE IMAGES")
        print("-" * 40)
        for i, f in enumerate(image_files, start=1):
            print(f"{i}. {f}")
        print("\nA. Process ALL images")
        choice = input("\nEnter image number or A: ").strip().lower()
        if choice == "a":
            selected = image_files
        else:
            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(image_files):
                    print("\nInvalid selection.")
                    return
                selected = [image_files[idx]]
            except ValueError:
                print("\nInvalid selection.")
                return

    for filename in selected:
        process_image(os.path.join(INPUT_DIR, filename), unet_model, clf,
                      roboflow_model, use_gemini=use_gemini)

    print("\n" + "=" * 65)
    print("ALL SELECTED IMAGES PROCESSED")
    print("=" * 65)
    print(f"\nResults are saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()