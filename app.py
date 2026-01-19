from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import base64
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from scipy.ndimage import label, center_of_mass
import os
import io

app = Flask(__name__)
CORS(app)

# ------------------ CONFIG ------------------
checkpoint_path = "eye_disease_unet_resnet18_multiscale_finetuned (1).pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ------------------ CLASSES & COLORS ------------------
classes = [
    "Background", "Central Serous Chorioretinopathy",
    "Diabetic Retinopathy", "Disc Edema", "Glaucoma", "Healthy",
    "Macular Scar", "Myopia", "Pterygium", "Retinal Detachment",
    "Retinitis Pigmentosa"
]
NUM_CLASSES = len(classes)
HEALTHY_IDX = classes.index("Healthy")

# warna RGB untuk overlay
COLORS = np.array([
    [0,0,0], [70,70,255], [255,200,0], 
    [255,0,180], [0,255,230], [160,0,0],
    [0,160,0], [11,106,160], [160,160,0], 
    [160,0,160], [0,160,160]
], dtype=np.uint8)

# ------------------ BUILD MODEL ------------------
def build_model(num_classes=NUM_CLASSES):
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
        encoder_depth=5
    )
    return model

model = build_model().to(DEVICE)

# ------------------ LOAD CHECKPOINT ------------------
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

raw_sd = torch.load(checkpoint_path, map_location=DEVICE)
if isinstance(raw_sd, dict) and ("model" in raw_sd or "state_dict" in raw_sd):
    if "model" in raw_sd:
        raw_sd = raw_sd["model"]
    elif "state_dict" in raw_sd:
        raw_sd = raw_sd["state_dict"]

new_sd = {}
for k, v in raw_sd.items():
    new_k = k
    if new_k.startswith("module."):
        new_k = new_k[len("module."):]
    if new_k.startswith("model."):
        new_k = new_k[len("model."):]
    new_sd[new_k] = v

try:
    model.load_state_dict(new_sd, strict=True)
except Exception as e:
    print("[WARN] strict load failed, trying non-strict:", e)
    model.load_state_dict(new_sd, strict=False)

model.eval()
print("[INFO] Model loaded and set to eval mode")

# ------------------ TRANSFORMS ------------------
infer_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ------------------ DICE COEFFICIENT ------------------
def dice_coefficient(mask1, mask2, eps=1e-7):
    """Hitung dice coefficient antara dua mask"""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    return (2. * intersection + eps) / (mask1.sum() + mask2.sum() + eps)

# ------------------ PREDICTION WITH CONSISTENCY ------------------
def predict_with_consistency(image_np, confidence_threshold=0.7):
    """
    Prediksi dengan menghitung consistency menggunakan pseudo dice
    Returns: mask, probabilities, consistency_score
    """
    H_orig, W_orig = image_np.shape[:2]

    # Transform & predict
    sample = infer_transform(image=image_np)
    tensor = sample["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)  # logits: 1, NUM_CLASSES, 256, 256
        prob = F.softmax(pred, dim=1)  # probabilitas
        mask_256 = torch.argmax(prob, dim=1).squeeze(0).cpu().numpy()
        
        # Simpan probability map untuk consistency check
        prob_map = prob.cpu().numpy()  # 1, NUM_CLASSES, 256, 256

    # Resize mask ke ukuran asli
    mask_orig = cv2.resize(mask_256.astype(np.uint8), (W_orig, H_orig), 
                           interpolation=cv2.INTER_NEAREST)
    
    # Resize probability map ke ukuran asli untuk setiap kelas
    prob_map_resized = np.zeros((NUM_CLASSES, H_orig, W_orig), dtype=np.float32)
    for c in range(NUM_CLASSES):
        prob_map_resized[c] = cv2.resize(prob_map[0, c], (W_orig, H_orig), 
                                         interpolation=cv2.INTER_LINEAR)
    
    return mask_orig, prob_map_resized

# ------------------ OVERLAY + STATS WITH CONSISTENCY ------------------
def make_overlay_and_stats(img_rgb, mask, prob_map, 
                          min_region_pixels=50, 
                          confidence_threshold=0.7):
    """
    Buat overlay dan statistik dengan consistency check
    """
    H, W = mask.shape

    # 1. Pisahkan background
    abnormal_pixels = mask[mask != 0]

    # Jika tidak ada piksel non-background → mata sehat
    if abnormal_pixels.size == 0:
        return img_rgb.copy(), [{
            "class": "Healthy",
            "percentage": 100.0,
            "confidence": 100.0
        }]

    # 2. Jika semua pixel abnormal = Healthy
    if np.all(abnormal_pixels == HEALTHY_IDX):
        return img_rgb.copy(), [{
            "class": "Healthy",
            "percentage": 100.0,
            "confidence": 100.0
        }]

    # 3. Hitung kelas dominan (tanpa Healthy)
    uniq, cnts = np.unique(abnormal_pixels, return_counts=True)

    # Hapus Healthy bila ada
    if HEALTHY_IDX in uniq:
        idx = np.where(uniq == HEALTHY_IDX)[0]
        uniq = np.delete(uniq, idx)
        cnts = np.delete(cnts, idx)

    # Jika tidak ada kelas abnormal tersisa
    if len(uniq) == 0:
        return img_rgb.copy(), [{
            "class": "Healthy",
            "percentage": 100.0,
            "confidence": 100.0
        }]

    # Ambil kelas dominan
    dom_class = int(uniq[np.argmax(cnts)])
    dom_name = classes[dom_class]

    # Mask untuk kelas dominan
    target_mask = (mask == dom_class).astype(np.uint8)

    # Jika jumlahnya terlalu kecil → anggap tidak valid
    if cnts.max() < min_region_pixels:
        return img_rgb.copy(), [{
            "class": "Healthy",
            "percentage": 100.0,
            "confidence": 100.0
        }]

    # ===== HITUNG PSEUDO DICE UNTUK CONSISTENCY =====
    confidence_map = prob_map[dom_class]  # H x W
    pseudo_mask = (confidence_map > confidence_threshold).astype(np.uint8)
    
    # Hitung dice coefficient
    pseudo_dice = dice_coefficient(target_mask, pseudo_mask)
    consistency_percent = pseudo_dice * 100

    # 4. Overlay hanya kelas dominan
    overlay = img_rgb.copy().astype(np.float32)
    disease_color = COLORS[dom_class].astype(np.float32)
    alpha = 0.45

    overlay[target_mask == 1] = (
        alpha * disease_color + (1 - alpha) * overlay[target_mask == 1]
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # 5. Contour
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    # 6. Statistik JSON dengan consistency
    total = len(abnormal_pixels)
    dominant_pixels = cnts.max()

    result_info = [{
        "class": dom_name,
        "pixels": int(dominant_pixels),
        "confidence": round(consistency_percent, 2)
    }]

    return overlay, result_info

# ------------------ ROUTES ------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})

@app.route("/predict", methods=["POST"])
def route_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form-data with key 'file'."}), 400
    
    file = request.files["file"]
    
    # Optional: confidence threshold dari request
    confidence_threshold = float(request.form.get("confidence_threshold", 0.7))
    
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_np = np.array(img)

        # Predict dengan consistency check
        mask, prob_map = predict_with_consistency(img_np, confidence_threshold)

        # Overlay + stats dengan consistency
        overlay, detected = make_overlay_and_stats(
            img_np, mask, prob_map, 
            min_region_pixels=200,
            confidence_threshold=confidence_threshold
        )

        # Encode to base64
        success, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not success:
            return jsonify({"error": "Failed to encode image"}), 500
        b64 = base64.b64encode(buf).decode("utf-8")

        return jsonify({
            "status": "success",
            "overlay": b64,
            "detected_labels": detected
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)