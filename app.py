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
checkpoint_path = "eye_disease_unet_resnet18_multiscale_finetuned (1).pth"  # ganti jika beda
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

# warna RGB untuk overlay (index 0 = background)
COLORS = np.array([
    [0,0,0], [70,70,255], [255,200,0], 
    [255,0,180], [0,255,230], [160,0,0],
    [0,160,0], [11,106,160], [160,160,0], 
    [160,0,160], [0,160,160]
], dtype=np.uint8)

# ------------------ BUILD MODEL ------------------
def build_model(num_classes=NUM_CLASSES):
    # jika environment mendukung, bisa pakai encoder_weights="imagenet"
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
        encoder_depth=5
    )
    return model

model = build_model().to(DEVICE)

# ------------------ LOAD CHECKPOINT (robust) ------------------
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

raw_sd = torch.load(checkpoint_path, map_location=DEVICE)
# Ambil key yang relevan bila checkpoint berbentuk dict
if isinstance(raw_sd, dict) and ("model" in raw_sd or "state_dict" in raw_sd):
    if "model" in raw_sd:
        raw_sd = raw_sd["model"]
    elif "state_dict" in raw_sd:
        raw_sd = raw_sd["state_dict"]

# Hilangkan prefix seperti "module." atau "model."
new_sd = {}
for k, v in raw_sd.items():
    new_k = k
    if new_k.startswith("module."):
        new_k = new_k[len("module."):]
    if new_k.startswith("model."):
        new_k = new_k[len("model."):]
    new_sd[new_k] = v

print("[INFO] Model loaded and set to eval mode")

# ------------------ PREDICTION (sama seperti COLAB) ------------------
infer_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def predict_single_scale(image_np):

    H_orig, W_orig = image_np.shape[:2]

    # Apply same transform that used in Colab
    sample = infer_transform(image=image_np)
    tensor = sample["image"].unsqueeze(0).to(DEVICE)  # 1,C,256,256

    with torch.no_grad():
        out = model(tensor)  # 1,NUM_CLASSES,256,256 (expected)
        mask_256 = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # 256x256

    # Resize mask kembali ke ukuran asli
    mask_orig = cv2.resize(mask_256.astype(np.uint8), (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
    return mask_orig

# ------------------ HELPER: overlay + labels (dengan filter small regions) ------------------
def make_overlay_and_stats(img_rgb, mask, min_region_pixels=50):

    H, W = mask.shape

    # ===============================
    # 1. Pisahkan background
    # ===============================
    abnormal_pixels = mask[mask != 0]

    # Jika tidak ada piksel non-background → mata sehat
    if abnormal_pixels.size == 0:
        return img_rgb.copy(), [{
            "class": "Healthy",
            "percentage": 100.0
        }]

    # ===============================
    # 2. Jika semua pixel abnormal = Healthy
    # ===============================
    HEALTHY_IDX = classes.index("Healthy")
    if np.all(abnormal_pixels == HEALTHY_IDX):
        return img_rgb.copy(), [{
            "class": "Healthy",
            "percentage": 100.0
        }]

    # ===============================
    # 3. Hitung kelas dominan (tanpa Healthy)
    # ===============================
    uniq, cnts = np.unique(abnormal_pixels, return_counts=True)

    # Hapus Healthy bila ada
    if HEALTHY_IDX in uniq:
        idx = np.where(uniq == HEALTHY_IDX)[0]
        uniq = np.delete(uniq, idx)
        cnts = np.delete(cnts, idx)

    # Ambil kelas dominan
    dom_class = int(uniq[np.argmax(cnts)])
    dom_name = classes[dom_class]

    # Mask hanya untuk kelas dominan
    target_mask = (mask == dom_class).astype(np.uint8)

    # Jika jumlahnya terlalu kecil → anggap tidak valid
    if cnts.max() < min_region_pixels:
        return img_rgb.copy(), [{
            "class": "Healthy",
            "percentage": 100.0
        }]

    # ===============================
    # 4. Overlay hanya kelas dominan
    # ===============================
    overlay = img_rgb.copy().astype(np.float32)
    disease_color = COLORS[dom_class].astype(np.float32)
    alpha = 0.45

    overlay[target_mask == 1] = (
        alpha * disease_color + (1 - alpha) * overlay[target_mask == 1]
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # ===============================
    # 5. Contour
    # ===============================
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    # ===============================
    # 6. Statistik JSON
    # ===============================
    total = len(abnormal_pixels)
    dominant_pixels = cnts.max()

    result_info = [{
        "class": dom_name,
        "pixels": int(dominant_pixels),
        "percentage": round(100 * dominant_pixels / total, 2)
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
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img_np = np.array(img)  # RGB

        # predict (single-scale seperti Colab)
        mask = predict_single_scale(img_np)

        # overlay + stats (dengan filter small regions to avoid label spam)
        overlay, detected = make_overlay_and_stats(img_np, mask, min_region_pixels=200)

        # encode to PNG base64
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
        # jangan expose stack trace sensitif di prod; ini untuk debugging lokal
        return jsonify({"error": str(e)}), 500

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
